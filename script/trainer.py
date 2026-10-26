#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeing Speech — shared trainer of the two training stages (entry points: main_1stage.py, main_2stage.py).

  stage 1 (main_1stage.py) : SAM pre-training.
            The encoder of the motion autoencoder (--axis_ae_ckpt) is copied into model.axis_vertex_encoder
            and frozen; everything else is trained with --lr.
            loss = velo_w*L_vel + recon_w*L_dir + 1e-6*L_store
  stage 2 (main_2stage.py) : TAC fine-tuning.
            The stage-1 checkpoint (--base_pretrained) is loaded with strict=False; all parameters are frozen
            except motion_integrator (--lr); with --lr_loaded > 0 the loaded backbone (except
            axis_vertex_encoder) is also fine-tuned at that LR.
            loss = L_recon + velo_w*L_vel + λ_lap*L_lap

Datasets    : --dataset vocaset (default) or TFHP (--period 25). dataloader/loader.py routes TFHP to
              dataloader/tfhp_loader.py (head-pose-free wo_vert, 100-frame windows, train hop 50).
              Both are FLAME topology -> same lip index (--lip_index) and template mesh (--template_ply).
Paths       : relative to the working directory (the run_bash/*.sh scripts cd to the repository root).
              --dataset_root  default data/<dataset>
              --template_ply  default <dataset_root>/FLAME_sample.ply
              --lip_index     default assets/lips_coordinates.npy
              --axis_ae_ckpt  default outputs/autoencoder/vocaset_d128/axis_ae_best.pth (run_bash/run_autoencoder.sh)
Validation  : after every epoch on the val split; the best epoch (lowest vertex MSE or lip LVE) is saved as
              <best>_model.pth and used for the predictions on the test split.
Checkpoints : <output_root>/<dataset>/save_<exp_name>/{<best>_model.pth, tmp_<e>_model.pth}   (<dataset> = vocaset | TFHP)
Predictions : <output_root>/<dataset>/result_<exp_name>/<seq>.npy   (test split, condition index 4)

Notes
  * --start_epoch N > 1 resumes from <output_root>/<dataset>/save_<exp>/N_model.pth and continues at epoch N+1.

Usage (from the repository root)
  python script/main_1stage.py --exp_name SeeingSpeech_VOCA_s1 [...]
  python script/main_2stage.py --exp_name SeeingSpeech_VOCA_s2 \
         --base_pretrained outputs/vocaset/save_SeeingSpeech_VOCA_s1/tmp_250_model.pth [...]
  python script/main_1stage.py --dataset TFHP --period 25 --exp_name SeeingSpeech_TFHP_s1 [...]
  see run_bash/run_vocaset.sh, run_bash/run_tfhp.sh
"""

import argparse
import os
import random
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

from dataloader.loader import get_dataloaders
from model.seeingspeech import SeeingSpeech


# =============================================================================
# Reproducibility / metrics / io
# =============================================================================
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def square_l2_error(gt, pred):
    """mean over (T, V_lip) of squared per-vertex L2 distance.  gt/pred: (..., T, V, 3)"""
    diff = gt.detach().cpu().numpy() - pred.detach().cpu().numpy()
    return float((np.linalg.norm(diff, axis=-1) ** 2).mean())


def ensure_dir(path, empty=False):
    if empty and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def load_any_state_dict(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Weight loading
# =============================================================================
def load_axis_ae_encoder(model, ckpt_path, freeze=True, required=False):
    """Copy the `encoder.*` weights of a motion-autoencoder checkpoint into model.axis_vertex_encoder."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        if required:   # stage 1: L_store needs the pretrained encoder
            raise SystemExit(f"[ERROR] motion-autoencoder ckpt not found: '{ckpt_path}' "
                             f"(train it with run_bash/run_autoencoder.sh or pass --axis_ae_ckpt)")
        print(f"[WARN] AxisAE ckpt not found: '{ckpt_path}' -> skip (stage 2 takes the encoder from --base_pretrained).")
        return

    print(f"[INFO] Loading AxisAE encoder from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    full_sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    enc_sd = {k[len("encoder."):]: v for k, v in full_sd.items() if k.startswith("encoder.")}
    if not enc_sd:
        raise SystemExit(f"[ERROR] no 'encoder.' weights in {ckpt_path} (expected a script/main_autoencoder.py checkpoint)")

    missing, unexpected = model.axis_vertex_encoder.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:   # e.g. an autoencoder with a different number of encoder layers
        raise SystemExit(f"[ERROR] {ckpt_path} does not match axis_vertex_encoder "
                         f"(missing={missing}, unexpected={unexpected})")
    print("[INFO] axis_vertex_encoder loaded.")
    if freeze:
        for p in model.axis_vertex_encoder.parameters():
            p.requires_grad = False
        print("[INFO] axis_vertex_encoder frozen.")


def load_base_pretrained(model, path, device):
    """Stage 2: load the stage-1 checkpoint (strict=False; motion_integrator.* stays at init)."""
    if not path or not os.path.exists(path):
        print(f"[WARN] base_pretrained not found: {path}")
        return set()
    print(f"[INFO] Loading base pretrained: {path}")
    sd = load_any_state_dict(path, map_location=device)
    loaded_keys = set(sd.keys()) & set(model.state_dict().keys())
    inc = model.load_state_dict(sd, strict=False)
    print(f"[INFO] load_state_dict: matched={len(loaded_keys)} missing={len(inc.missing_keys)} "
          f"unexpected={len(inc.unexpected_keys)}")
    return loaded_keys


# =============================================================================
# Optimizer per stage
# =============================================================================
def build_optimizer(args, model, loaded_keys):
    if args.stage == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"[INFO] stage 1: trainable params = {sum(p.numel() for p in params):,} @ lr={args.lr}")
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # stage 2: freeze everything, unfreeze TAC, optionally the loaded backbone (never axis_vertex_encoder)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.motion_integrator.parameters():
        p.requires_grad = True
    integrator_ids = {id(p) for p in model.motion_integrator.parameters()}
    integrator_params = list(model.motion_integrator.parameters())

    loaded_params = []
    if args.lr_loaded > 0.0:
        for name, p in model.named_parameters():
            if id(p) in integrator_ids or name.startswith("axis_vertex_encoder."):
                continue
            if name in loaded_keys:
                p.requires_grad = True
                loaded_params.append(p)

    n_int = sum(p.numel() for p in integrator_params)
    if loaded_params:
        n_ld = sum(p.numel() for p in loaded_params)
        print(f"[INFO] stage 2 dual-LR: integrator {n_int:,} @ lr={args.lr} | loaded backbone {n_ld:,} "
              f"@ lr={args.lr_loaded} | axis_vertex_encoder frozen")
        groups = [{"params": integrator_params, "lr": args.lr},
                  {"params": loaded_params, "lr": args.lr_loaded}]
        return torch.optim.Adam(groups, weight_decay=args.weight_decay)
    print(f"[INFO] stage 2 integrator-only: trainable params = {n_int:,} @ lr={args.lr}")
    return torch.optim.Adam(integrator_params, lr=args.lr, weight_decay=args.weight_decay)


# =============================================================================
# Train / test
# =============================================================================
def set_train_mode(model, args):
    """train() only the modules that are optimised; frozen ones run in eval mode (no dropout / HuBERT masking).
    stage 1: everything except the frozen motion-autoencoder encoder (deterministic L_store targets).
    stage 2: only TAC (motion_integrator); with --lr_loaded > 0 also the loaded backbone."""
    if args.stage == 2 and args.lr_loaded <= 0.0:
        model.eval()
        model.motion_integrator.train()
    else:
        model.train()
    model.axis_vertex_encoder.eval()


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, lip):
    save_path = args.save_path
    if args.start_epoch <= 1:
        ensure_dir(save_path, empty=True)
        first_epoch, iteration = 1, 0
    else:
        ensure_dir(save_path, empty=False)
        resume = os.path.join(save_path, f"{args.start_epoch}_model.pth")
        print(f"[INFO] Resume from: {resume}")
        model.load_state_dict(load_any_state_dict(resume, map_location=args.device), strict=False)
        first_epoch, iteration = args.start_epoch + 1, len(train_loader) * args.start_epoch

    # slot weights: stage 1 -> [1, velo_w, recon_w, 1, 1, 1] ; stage 2 -> [1, velo_w, 1, 1, 1, 1]  (slot 2 = L_vel)
    slot_w = [1.0, args.velo_loss_weight, args.recon_loss_weight, 1.0, 1.0, 1.0] if args.stage == 1 \
        else [1.0, args.velo_loss_weight, 1.0, 1.0, 1.0, 1.0]
    names = model.loss_names

    best_epoch = max(1, args.start_epoch)
    best_loss, best_lve = np.inf, np.inf
    for e in range(first_epoch, args.max_epoch + 1):
        set_train_mode(model, args)
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (audio, vertice, template, one_hot, _) in pbar:
            iteration += 1
            audio, vertice, template, one_hot = (t.to(args.device) for t in (audio, vertice, template, one_hot))

            _, *slots = model(audio, template, vertice, one_hot, inference=False)
            loss = sum(w * s for w, s in zip(slot_w, slots))
            loss.backward()
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            desc = " ".join(f"{n}:{float(s.item()) * 1e5:.3f}" for n, s in zip(names, slots) if n != "-")
            pbar.set_description(f"(E{e} it{iteration}) {desc} TOTAL:{float(loss.item()) * 1e5:.3f}")

        # ---- validation on the val split (first speaker condition) ----
        model.eval()
        v_loss, v_lve = [], []
        with torch.no_grad():
            for audio, vertice, template, one_hot_all, _ in dev_loader:
                if torch.all(one_hot_all.squeeze() == 0):
                    continue
                audio, vertice, template, one_hot_all = (t.to(args.device) for t in (audio, vertice, template, one_hot_all))
                est, *_ = model(audio, template, vertice, one_hot_all[:, 0, :], inference=True)
                T = min(est.size(1), vertice.size(1))
                est, gt = est[:, :T], vertice[:, :T]
                v_loss.append(float(criterion(est, gt).item()))
                V = args.vertice_dim // 3
                v_lve.append(square_l2_error(gt.reshape(-1, T, V, 3)[:, :, lip], est.reshape(-1, T, V, 3)[:, :, lip]))
        cur_loss = float(np.mean(v_loss)) if v_loss else np.inf
        cur_lve = float(np.mean(v_lve)) if v_lve else np.inf

        improved = False
        if cur_lve < best_lve:
            best_lve, improved = cur_lve, True
        if cur_loss < best_loss:
            best_loss, improved = cur_loss, True
        if improved:
            best_epoch = e
            torch.save(model.state_dict(), os.path.join(save_path, f"{best_epoch}_model.pth"))
        if e % 25 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"tmp_{e}_model.pth"))
        print(f"[VAL] epoch:{e} | v_mse:{cur_loss * 1e5:.5f} | lip_lve:{cur_lve * 1e5:.5f} | best:{best_epoch}")
    return model, best_epoch


@torch.no_grad()
def test(args, model, test_loader, epoch, condition_idx=4):
    result_path = args.result_path
    ensure_dir(result_path, empty=True)
    ckpt = os.path.join(args.save_path, f"{epoch}_model.pth")
    print(f"[INFO] Testing with: {ckpt}")
    model.load_state_dict(load_any_state_dict(ckpt, map_location=args.device), strict=False)
    model.to(args.device).eval()

    for audio, vertice, template, one_hot_all, file_name in test_loader:
        if torch.all(one_hot_all.squeeze() == 0):
            continue
        audio, vertice, template, one_hot_all = (t.to(args.device) for t in (audio, vertice, template, one_hot_all))
        idx = condition_idx if condition_idx < one_hot_all.shape[1] else 0   # fewer train speakers than 5
        pred, *_ = model(audio, template, vertice, one_hot_all[:, idx, :], inference=True)
        np.save(os.path.join(result_path, file_name[0].split(".")[0] + ".npy"),
                pred.squeeze().detach().cpu().numpy())
    print(f"[INFO] predictions saved -> {result_path}")


# =============================================================================
# Main
# =============================================================================
def build_parser(stage):
    task = "SAM pre-training" if stage == 1 else "TAC fine-tuning"
    p = argparse.ArgumentParser(description=f"Seeing Speech — stage {stage} ({task}), VOCASET / TFHP")
    p.add_argument("--exp_name", type=str, required=True, help="save_<exp_name> / result_<exp_name>")
    p.add_argument("--start_epoch", type=int, default=1, help="1: from scratch; N>1: resume from N_model.pth")
    p.add_argument("--test_only", action="store_true", help="skip training, run test() with --start_epoch ckpt")

    # data (VOCASET defaults; TFHP: --dataset TFHP --period 25, see run_bash/run_tfhp.sh)
    p.add_argument("--dataset", type=str, default="vocaset", choices=["vocaset", "TFHP"])
    p.add_argument("--vertice_dim", type=int, default=5023 * 3)
    p.add_argument("--identities_class", type=int, default=8)
    p.add_argument("--period", type=int, default=30)
    p.add_argument("--wav_path", type=str, default="wav")
    p.add_argument("--vertices_path", type=str, default="vertices_npy")
    p.add_argument("--template_file", type=str, default="templates.pkl")
    p.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                   " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                   " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                   " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    p.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA")
    p.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
    p.add_argument("--dataset_root", type=str, default="", help="empty: data/<dataset>")
    p.add_argument("--template_ply", type=str, default="", help="FLAME mesh for the TAC graph; empty: <dataset_root>/FLAME_sample.ply")
    p.add_argument("--lip_index", type=str, default="assets/lips_coordinates.npy", help="FLAME lip vertex indices (validation LVE)")
    p.add_argument("--output_root", type=str, default="outputs", help="checkpoints / predictions root")
    # TFHP only (speaker splits come from <dataset_root>/{train,val,test}.txt; the *_subjects above are ignored)
    p.add_argument("--tfhp_min_len", type=int, default=0, help="TFHP: skip clips with T <= this (0: keep all)")
    p.add_argument("--batch_size", type=int, default=1, help="train batch size (val/test always 1)")
    p.add_argument("--num_workers", type=int, default=4, help="TFHP train loader workers")

    # model
    p.add_argument("--feature_dim", type=int, default=128)
    p.add_argument("--mel_feature_dim", type=int, default=128)
    p.add_argument("--mem_slot", type=int, default=32)
    p.add_argument("--mem_win", type=int, default=15,
                   help="memory window length 2Ω+1 (paper Ω=7 -> 15); must match the stage-1 ckpt in stage 2")
    p.add_argument("--softmax_scaling_term", type=float, default=16.0)
    p.add_argument("--axis_mixer_dim", type=int, default=64, help="D_TAC (GCN hidden dim)")
    p.add_argument("--axis_ae_ckpt", type=str, default="outputs/autoencoder/vocaset_d128/axis_ae_best.pth",
                   help="motion-autoencoder checkpoint (script/main_autoencoder.py)")
    p.add_argument("--base_pretrained", type=str, default="", help="stage 2: stage-1 checkpoint")

    # optimisation
    p.add_argument("--lr", type=float, default=1e-4, help="stage 1: all trainable; stage 2: motion_integrator")
    p.add_argument("--lr_loaded", type=float, default=0.0, help="stage 2: >0 also fine-tunes the loaded backbone")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_epoch", type=int, default=250 if stage == 1 else 150, help="paper: stage 1 250, stage 2 150")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    # loss weights
    p.add_argument("--velo_loss_weight", type=float, default=1.0, help="L_vel weight (stage 1: per direction, stage 2: composed)")
    p.add_argument("--recon_loss_weight", type=float, default=1.0, help="stage 1: L_dir weight")
    p.add_argument("--lap_loss_weight", type=float, default=1e3, help="stage 2: λ_lap")
    return p


def run(stage):
    set_seed(0)
    args = build_parser(stage).parse_args()
    args.stage = stage
    args.dataset_root = args.dataset_root or os.path.join("data", args.dataset)
    args.template_ply = args.template_ply or os.path.join(args.dataset_root, "FLAME_sample.ply")
    args.save_path = os.path.join(args.output_root, args.dataset, f"save_{args.exp_name}")
    args.result_path = os.path.join(args.output_root, args.dataset, f"result_{args.exp_name}")
    if args.stage == 2 and not args.base_pretrained:
        raise SystemExit("--base_pretrained (stage-1 checkpoint) is required for stage 2")
    lip = np.load(args.lip_index)

    print(f"[INFO] stage={args.stage} exp={args.exp_name} d={args.feature_dim} md={args.mel_feature_dim} "
          f"s={args.mem_slot} win={args.mem_win} period={args.period}")
    model = SeeingSpeech(args)
    load_axis_ae_encoder(model, args.axis_ae_ckpt, freeze=True, required=(args.stage == 1))
    loaded_keys = load_base_pretrained(model, args.base_pretrained, args.device) if args.stage == 2 else set()
    model = model.to(args.device)
    optimizer = build_optimizer(args, model, loaded_keys)
    print(f"[INFO] trainable parameters: {count_parameters(model):,}")

    dataset = get_dataloaders(args)
    criterion = nn.MSELoss()

    if args.test_only:
        test(args, model, dataset["test"], epoch=args.start_epoch)
        return
    model, best_epoch = trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion, lip)
    test(args, model, dataset["test"], epoch=best_epoch)
