#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeing Speech — motion autoencoder pre-training (paper App. B.1, Table S1).

  model : model/autoencoder.py (MotionAutoEncoder); the best checkpoint (axis_ae_best.pth, selected on the
          val split) is the --axis_ae_ckpt of main_1stage.py / main_2stage.py
  data  : --datasets vocaset (default) | tfhp | vocaset,tfhp (batches drawn from both)
  input : per-direction motion (B,T,V), z-scored per vertex along time
  output: reconstruction un-normalised back to the raw motion domain
  loss  : Eq. S1, in the raw domain: loss_recon + lambda_velo * loss_velo
          loss_recon = MSE(pred, gt), loss_velo = MSE(Δpred, Δgt) (first-order temporal difference)

  python script/main_autoencoder.py --save_tag vocaset_d128     (see run_bash/run_autoencoder.sh)
"""

import argparse
import copy
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # package root

from dataloader.loader import get_dataloaders as get_voca_dataloaders
from dataloader.tfhp_loader import get_dataloaders as get_tfhp_dataloaders
from model.autoencoder import MotionAutoEncoder


# =========================
# Seed
# =========================
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)


# =========================
# Joint train loader: each step draws a batch from a randomly chosen dataset
# =========================
class JointTrainLoader:
    def __init__(self, loaders_dict, steps_per_epoch=None):
        self.ds_names = list(loaders_dict.keys())
        self.loaders = [loaders_dict[n] for n in self.ds_names]
        self.lengths = [len(l) for l in self.loaders]

        if steps_per_epoch is None:
            self.steps_per_epoch = sum(self.lengths)
        else:
            self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        iters = [iter(l) for l in self.loaders]

        for _ in range(self.steps_per_epoch):
            k = random.randrange(len(self.loaders))
            name = self.ds_names[k]

            try:
                batch = next(iters[k])
            except StopIteration:
                iters[k] = iter(self.loaders[k])
                batch = next(iters[k])

            yield name, batch


# =========================
# Data loaders (train: joint loader; validation: the val split of each dataset)
# =========================
def build_joint_dataloaders(args):
    names = [s.strip().lower() for s in args.datasets.split(",")]
    unknown = sorted(set(names) - {"vocaset", "tfhp"})
    if unknown:
        raise ValueError(f"Unknown --datasets entries {unknown} (use vocaset, tfhp or vocaset,tfhp)")

    loaders_train = {}
    loaders_dev = {}

    if "vocaset" in names:
        voca_args = copy.deepcopy(args)
        voca_args.dataset = "vocaset"
        voca_args.dataset_root = args.vocaset_root
        voca_ds = get_voca_dataloaders(voca_args)
        loaders_train["vocaset"] = voca_ds["train"]
        loaders_dev["vocaset"] = voca_ds["valid"]

    if "tfhp" in names:
        tfhp_args = copy.deepcopy(args)
        tfhp_args.dataset = "TFHP"
        tfhp_args.dataset_root = args.tfhp_root
        tfhp_ds = get_tfhp_dataloaders(tfhp_args)
        loaders_train["TFHP"] = tfhp_ds["train"]
        loaders_dev["TFHP"] = tfhp_ds["valid"]

    if len(loaders_train) == 0:
        raise ValueError(f"No dataset selected. args.datasets={args.datasets}")

    joint_train_loader = JointTrainLoader(loaders_train)
    return joint_train_loader, loaders_dev


def parse_batch(batch, dataset_name, device):
    if dataset_name in ["vocaset", "TFHP"]:
        audio, vertice, template, one_hot, file_name = batch
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return vertice.to(device), template.to(device)


# =========================
# Trainer
# =========================
def trainer_joint(
    args,
    joint_train_loader,
    dev_loaders_dict,
    model,
    optimizer,
    epoch=300,
    input_epoch=1,
):
    save_dir = os.path.join(args.output_root, "autoencoder", args.save_tag)
    if input_epoch == 1 and args.reset_save_dir:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    iteration = 0
    best_val = np.inf
    best_epoch = input_epoch

    device = torch.device(args.device)
    V = args.vertice_dim // 3

    VERTEX_WEIGHT = 1.0
    lambda_velo = float(args.lambda_velo)

    print(f"[INFO] V={V}, num_layers={args.num_layers} (encoder) + {args.num_layers} (decoder), "
          f"lambda_velo={lambda_velo}")

    # per-vertex z-score along time (input only); the stats un-normalise the output.
    # vertices that do not move (std ~ 0) are clamped to min_std, so their normalised input stays ~0.
    def motion_zscore_with_stats(x_btv_raw, eps=1e-8, min_std=1e-5):
        """
        x_btv_raw : (B,T,V) raw motion
        return:
          x_norm: (B,T,V)
          mean : (B,1,V)
          std  : (B,1,V)
        """
        mean = x_btv_raw.mean(dim=1, keepdim=True)                            # (B,1,V)
        std  = x_btv_raw.std(dim=1, keepdim=True, unbiased=False)             # (B,1,V)
        std  = torch.clamp(std, min=min_std)
        x_norm = (x_btv_raw - mean) / std
        return x_norm, mean, std

    def unnormalize(pred_norm, mean_btv, std_btv):
        """
        pred_norm: (B,T,V)
        mean_btv/std_btv: (B,1,V)
        return: pred_raw (B,T,V)
        """
        return pred_norm * std_btv + mean_btv

    def velo_mse(pred_btv, gt_btv):
        """MSE of the first-order temporal difference; pred/gt: (B,T,V); 0 if T < 2."""
        if pred_btv.shape[1] < 2:
            return torch.zeros((), device=pred_btv.device)
        dp = pred_btv[:, 1:] - pred_btv[:, :-1]
        dg = gt_btv[:, 1:]  - gt_btv[:, :-1]
        return F.mse_loss(dp, dg)

    def forward_and_losses(gt_x_raw, gt_y_raw, gt_z_raw):
        """return: loss_recon, loss_velo (averaged over the three directions)"""
        in_x_norm, mean_x, std_x = motion_zscore_with_stats(gt_x_raw)
        in_y_norm, mean_y, std_y = motion_zscore_with_stats(gt_y_raw)
        in_z_norm, mean_z, std_z = motion_zscore_with_stats(gt_z_raw)

        x_pred_norm, y_pred_norm, z_pred_norm = model(in_x_norm, in_y_norm, in_z_norm)
        x_pred_raw = unnormalize(x_pred_norm, mean_x, std_x)
        y_pred_raw = unnormalize(y_pred_norm, mean_y, std_y)
        z_pred_raw = unnormalize(z_pred_norm, mean_z, std_z)

        loss_rx = F.mse_loss(x_pred_raw, gt_x_raw)
        loss_ry = F.mse_loss(y_pred_raw, gt_y_raw)
        loss_rz = F.mse_loss(z_pred_raw, gt_z_raw)
        loss_recon = (loss_rx + loss_ry + loss_rz) / 3.0

        loss_vx = velo_mse(x_pred_raw, gt_x_raw)
        loss_vy = velo_mse(y_pred_raw, gt_y_raw)
        loss_vz = velo_mse(z_pred_raw, gt_z_raw)
        loss_velo = (loss_vx + loss_vy + loss_vz) / 3.0

        return loss_recon, loss_velo

    for e in range(input_epoch, epoch + 1):
        model.train()
        recon_log, velo_log = [], []

        pbar = tqdm(enumerate(joint_train_loader), total=len(joint_train_loader))
        optimizer.zero_grad()

        for i, (ds_name, batch) in pbar:
            iteration += 1

            vertice, template = parse_batch(batch, ds_name, device)
            B = template.shape[0]
            template = template.unsqueeze(1)  # (B,1,V*3)

            gt_motion = (vertice - template) * VERTEX_WEIGHT
            gt_motion_reshaped = gt_motion.reshape(B, -1, V, 3)

            gt_x_raw = gt_motion_reshaped[:, :, :, 0]  # (B,T,V)
            gt_y_raw = gt_motion_reshaped[:, :, :, 1]
            gt_z_raw = gt_motion_reshaped[:, :, :, 2]

            B_cur, T_cur, V_cur = gt_x_raw.shape
            assert V_cur == V

            loss_recon, loss_velo = forward_and_losses(gt_x_raw, gt_y_raw, gt_z_raw)
            loss = loss_recon + lambda_velo * loss_velo

            loss.backward()
            recon_log.append(float(loss_recon.detach().item()))
            velo_log.append(float(loss_velo.detach().item()))

            if (i + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description(
                f"(E{e} it{iteration}) DS:{ds_name} "
                f"R:{loss_recon.item()*1e5:.3f} V:{loss_velo.item()*1e5:.3f} "
                f"Tot:{loss.item()*1e5:.3f} (x1e-5)"
            )

        if len(recon_log) > 0:
            print(
                f"[TRAIN] Epoch {e}, "
                f"Recon: {np.mean(recon_log)*1e5:.4f}, "
                f"Velo:  {np.mean(velo_log)*1e5:.4f}  (x1e-5)"
            )

        # ---- validation (val split, recon + velo in the raw domain) ----
        model.eval()
        val_recon_all, val_velo_all = [], []

        with torch.no_grad():
            for ds_name, dev_loader in dev_loaders_dict.items():
                ds_recon_log, ds_velo_log = [], []
                for batch in dev_loader:
                    vertice, template = parse_batch(batch, ds_name, device)

                    B = template.shape[0]
                    template = template.unsqueeze(1)

                    gt_motion = (vertice - template) * VERTEX_WEIGHT
                    gt_motion_reshaped = gt_motion.reshape(B, -1, V, 3)

                    gt_x_raw = gt_motion_reshaped[:, :, :, 0]
                    gt_y_raw = gt_motion_reshaped[:, :, :, 1]
                    gt_z_raw = gt_motion_reshaped[:, :, :, 2]

                    loss_recon, loss_velo = forward_and_losses(gt_x_raw, gt_y_raw, gt_z_raw)

                    ds_recon_log.append(float(loss_recon.item()))
                    ds_velo_log.append(float(loss_velo.item()))
                    val_recon_all.append(float(loss_recon.item()))
                    val_velo_all.append(float(loss_velo.item()))

                if len(ds_recon_log) > 0:
                    print(
                        f"[VAL-{ds_name}] Epoch {e}, "
                        f"Recon: {np.mean(ds_recon_log)*1e5:.4f} (x1e-5), "
                        f"Velo:  {np.mean(ds_velo_log)*1e5:.4f} (x1e-5)"
                    )

        current_recon = float(np.mean(val_recon_all)) if len(val_recon_all) > 0 else 0.0
        current_velo  = float(np.mean(val_velo_all))  if len(val_velo_all)  > 0 else 0.0
        current_total = current_recon + lambda_velo * current_velo
        print(
            f"[VAL-ALL] Epoch {e}, "
            f"Recon: {current_recon*1e5:.4f} (x1e-5), "
            f"Velo:  {current_velo*1e5:.4f} (x1e-5), "
            f"Total: {current_total*1e5:.4f} (x1e-5)"
        )

        if current_total <= best_val:
            best_val = current_total
            best_epoch = e

            torch.save(
                {
                    "epoch": best_epoch,
                    "best_val_total": best_val,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "num_layers": args.num_layers,
                    "lambda_velo": lambda_velo,
                },
                os.path.join(save_dir, "axis_ae_best.pth")
            )
            print(f"[SAVE] Best model updated at epoch {best_epoch}, total={best_val:.7f}")

        if e % 50 == 0:
            torch.save(
                {
                    "epoch": e,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                },
                os.path.join(save_dir, f"axis_ae_tmp_{e}.pth")
            )

    return model, best_epoch


# =========================
# main
# =========================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    input_epoch = 1

    parser = argparse.ArgumentParser(description="Seeing Speech — motion autoencoder pre-training")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vertice_dim", type=int, default=5023*3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=250, help="paper: 250")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1, help="paper: 1")

    parser.add_argument("--save_tag", type=str, required=True,
                        help="output dir: <output_root>/autoencoder/<save_tag>/axis_ae_best.pth")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--reset_save_dir", action="store_true", help="empty the output dir before training")

    # Transformer layers (paper Table S1)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Transformer layers in the encoder and in the decoder (paper: 1 + 1)")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--trans_dropout", type=float, default=0.1)
    parser.add_argument("--ff_mult", type=float, default=4.0)

    parser.add_argument("--lambda_velo", type=float, default=1.0,
                        help="velocity loss weight (MSE on 1st-order temporal diff)")
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument("--datasets", type=str, default="vocaset", help="vocaset | tfhp | vocaset,tfhp")

    # VOCASET
    parser.add_argument("--vocaset_root", type=str, default="data/vocaset")
    parser.add_argument("--identities_class", type=int, default=8)
    parser.add_argument("--wav_path", type=str, default="wav")
    parser.add_argument("--vertices_path", type=str, default="vertices_npy")
    parser.add_argument("--template_file", type=str, default="templates.pkl")
    parser.add_argument("--train_subjects", type=str,
                        default="FaceTalk_170728_03272_TA"
                                " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str,
                        default="FaceTalk_170811_03275_TA"
                                " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str,
                        default="FaceTalk_170809_00138_TA"
                                " FaceTalk_170731_00024_TA")

    # TFHP (speaker splits from <tfhp_root>/{train,val,test}.txt)
    parser.add_argument("--tfhp_root", type=str, default="data/TFHP")
    parser.add_argument("--tfhp_min_len", type=int, default=99, help="skip clips with T <= this (0: keep all)")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    device = torch.device(args.device)

    V = args.vertice_dim // 3
    model = MotionAutoEncoder(
        V=V,
        feature_dim=args.feature_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.trans_dropout,
        ff_mult=args.ff_mult,
    ).to(device)

    print("Motion autoencoder parameters: ", count_parameters(model))

    joint_train_loader, dev_loaders_dict = build_joint_dataloaders(args)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    model, best_epoch = trainer_joint(
        args, joint_train_loader, dev_loaders_dict,
        model, optimizer,
        epoch=args.max_epoch, input_epoch=input_epoch
    )

    save_dir = os.path.join(args.output_root, "autoencoder", args.save_tag)
    print(f"[DONE] Pre-training finished. Best epoch: {best_epoch}")
    print(f"[INFO] Best model saved to: {os.path.join(save_dir, 'axis_ae_best.pth')}")


if __name__ == "__main__":
    main()
