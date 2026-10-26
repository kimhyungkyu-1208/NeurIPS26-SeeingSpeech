#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeing Speech — model definition for both training stages (class SeeingSpeech).

Paper naming:
  SAM  = Speech–Articulatory Memory
         MelspecEncoder + MemoryNetwork (M_key / M_val,δ / contextual query head / w_δ)
         + content_mlp + PPE + per-direction transformer decoders g_δ + vertice_{x,y,z}_map_r
  TAC  = Topology-aware Articulatory Composition
         GraphMotionIntegrator (GCN→TCN ×2 + region-adaptive gate)      [stage 2 only]
  L_store reference features come from axis_vertex_encoder (frozen encoder of the motion autoencoder in
  model/autoencoder.py, training only).

Stage selection: `args.stage` (1 or 2).
  stage 1 : SAM only.       loss = L_dir + L_vel + 1e-6 * L_store         (per-direction)
  stage 2 : SAM + TAC.      loss = L_recon(MSE) + L_vel + λ_lap * L_lap    (on composed motion)
            SAM / AxisAE weights are loaded from the stage-1 checkpoint and frozen by the trainer.

Return value (7-tuple):
  (vertice_out, slot1, slot2, slot3, slot4, slot5, slot6)
    stage 1 : (v, 0,        L_vel,   L_dir,  1e-6*L_store, 0, 0)
    stage 2 : (v, L_recon,  L_vel,   L_lap,  0,            0, 0)
  `SeeingSpeech.loss_names` gives the human-readable name of each slot for the current stage.
"""

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms
import trimesh

from .hubert import HubertModel


# =============================================================================
# Utils
# =============================================================================
def linear_interpolation(features, output_len):
    """(B,T,D) -> (B,output_len,D) linear resampling along time."""
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


class PeriodicPositionalEncoding(nn.Module):
    """Periodic PE for the content (decoder tgt) stream."""
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal PE (same as the motion autoencoder)."""
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def _build_transformer_encoder(feature_dim, num_heads, ff_mult, dropout, num_layers):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=feature_dim,
        nhead=num_heads,
        dim_feedforward=int(feature_dim * ff_mult),
        dropout=dropout,
        batch_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


# =============================================================================
# Motion encoder (same layers as the encoder in model/autoencoder.py; provides L_store targets)
# =============================================================================
class AxisVertexTransformerEncoder(nn.Module):
    """(B,T,V) per axis -> Linear(V->d) -> PE -> TransformerEncoder -> (B,T,d), axes fully separated."""
    def __init__(self, V, feature_dim, num_layers=2, num_heads=4, ff_mult=4.0,
                 dropout=0.1, use_axis_embedding=False):
        super().__init__()
        self.V = V
        self.feature_dim = feature_dim
        self.use_axis_embedding = bool(use_axis_embedding)

        self.enc_x_in = nn.Linear(V, feature_dim)
        self.enc_y_in = nn.Linear(V, feature_dim)
        self.enc_z_in = nn.Linear(V, feature_dim)

        self.pos_encoding = PositionalEncoding(d_model=feature_dim, dropout=dropout)

        self.transformer_x = _build_transformer_encoder(feature_dim, num_heads, ff_mult, dropout, num_layers)
        self.transformer_y = _build_transformer_encoder(feature_dim, num_heads, ff_mult, dropout, num_layers)
        self.transformer_z = _build_transformer_encoder(feature_dim, num_heads, ff_mult, dropout, num_layers)

        if self.use_axis_embedding:
            self.axis_embed = nn.Embedding(3, feature_dim)
            nn.init.normal_(self.axis_embed.weight, mean=0.0, std=0.02)
        else:
            self.axis_embed = None

        for m in [self.enc_x_in, self.enc_y_in, self.enc_z_in]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def _encode_axis(self, axis_in, enc_in, transformer, axis_id):
        h = enc_in(axis_in)
        if self.use_axis_embedding:
            e = self.axis_embed(torch.tensor(axis_id, device=h.device)).view(1, 1, -1)
            h = h + e
        h = self.pos_encoding(h)
        return transformer(h)

    def encode_axes(self, x_in, y_in, z_in):
        h_x = self._encode_axis(x_in, self.enc_x_in, self.transformer_x, axis_id=0)
        h_y = self._encode_axis(y_in, self.enc_y_in, self.transformer_y, axis_id=1)
        h_z = self._encode_axis(z_in, self.enc_z_in, self.transformer_z, axis_id=2)
        return h_x, h_y, h_z


# =============================================================================
# Basic blocks
# =============================================================================
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            self.safe_norm(cout),
        )
        self.act = nn.LeakyReLU()
        self.residual = residual

    def safe_norm(self, c_out):
        g = max(1, c_out // 16)
        while c_out % g != 0:
            g -= 1
        return nn.GroupNorm(g, c_out)

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class FC3(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)


class FC1(nn.Module):
    def __init__(self, in_dim, out_dim, is_encoder=True):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        if is_encoder:
            nn.init.xavier_uniform_(self.layer.weight)
        else:
            nn.init.normal_(self.layer.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.layer.bias, 0.0)

    def forward(self, x):
        return self.layer(x)


# =============================================================================
# SAM (1/3): acoustic encoder — mel spectrogram -> 2D conv stack adapted from the Wav2Lip audio encoder
#   (Prajwal et al., "A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild", ACM MM 2020,
#    https://github.com/Rudrabha/Wav2Lip). Changes: GroupNorm + LeakyReLU instead of BatchNorm + ReLU,
#    different strides, frequency average + linear projection to mel_feature_dim.
# =============================================================================
class MelExtractor(nn.Module):
    def __init__(self, sr=16000, n_fft=200, win_length=200, hop_length=80, n_mels=40):
        super().__init__()
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, n_mels=n_mels,
        )

    def forward(self, audio_tensor):
        return self.mel_transform(audio_tensor)


class MelspecEncoder(nn.Module):
    def __init__(self, sr=16000, n_fft=1024, win_length=1024, hop_length=256,
                 n_mels=128, mel_feature_dim=64):
        super().__init__()
        self.mel_extractor = MelExtractor(
            sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels,
        )
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )
        self.proj = nn.Linear(512, mel_feature_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)
        for layer in self.audio_encoder:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, audio):
        mel_spec = self.mel_extractor(audio).unsqueeze(1)   # (B,1,n_mels,T)
        feat = self.audio_encoder(mel_spec)                 # (B,512,F',T')
        feat = feat.permute(0, 3, 1, 2).contiguous()        # (B,T,512,F')
        feat = feat.mean(dim=-1)                            # (B,T,512)
        return self.proj(feat)                              # (B,T,mel_feature_dim)


# =============================================================================
# SAM (2/3): memory addressing + directional retrieval + L_store
# =============================================================================
class MemoryNetwork(nn.Module):
    """
    mel feature -> contextual query (Conv1d over a window of W = 2Ω+1 frames, W = args.mem_win)
    -> cosine addressing over M_key (Eq. 1) -> per-direction retrieval from M_val,δ ∈ R^{N×(2Ω+1)×d} (Eq. 2)
    -> learnable temporal weighting w_δ.
    aux_loss = L_store (Eq. 5): cosine distance between retrieved (2Ω+1)-frame windows and GT motion-feature windows.
    """
    def __init__(self, args):
        super().__init__()
        self.motion_vertice_dim = args.vertice_dim
        self.feature_dim = args.feature_dim
        self.mel_feature_dim = args.mel_feature_dim
        self.mem_slot = args.mem_slot
        self.device = args.device
        self.dataset = args.dataset
        self.scaling_term = args.softmax_scaling_term
        self.win = getattr(args, "mem_win", 15)

        # M_key
        self.mel_memory = nn.Parameter(torch.Tensor(self.mem_slot, self.mel_feature_dim), requires_grad=True)
        nn.init.xavier_normal_(self.mel_memory)

        self.mel_encoder = FC1(self.mel_feature_dim, self.mel_feature_dim)
        self.drop = nn.Dropout(0.0)

        # contextual query head over Ω frames
        self.audio2query = nn.Conv1d(
            self.mel_feature_dim, self.mel_feature_dim,
            kernel_size=self.win, padding=self.win // 2, bias=True,
        )
        nn.init.xavier_uniform_(self.audio2query.weight)
        nn.init.constant_(self.audio2query.bias, 0)

        # M_val,δ  (δ = x:S, y:O, z:P)
        self.motion_x_memory_seq = nn.Parameter(torch.randn(self.mem_slot, self.win, self.feature_dim))
        self.motion_y_memory_seq = nn.Parameter(torch.randn(self.mem_slot, self.win, self.feature_dim))
        self.motion_z_memory_seq = nn.Parameter(torch.randn(self.mem_slot, self.win, self.feature_dim))

        # w_δ(ω)
        self.level_x_w = nn.Parameter(torch.full((self.win,), 1.0 / self.win))
        self.level_y_w = nn.Parameter(torch.full((self.win,), 1.0 / self.win))
        self.level_z_w = nn.Parameter(torch.full((self.win,), 1.0 / self.win))

    def f2msim(self, feature, memory):
        norm_f = F.normalize(feature, p=2, dim=-1)
        norm_m = F.normalize(memory, p=2, dim=-1)
        sim = torch.einsum("bfd,sd->bfs", norm_f, norm_m)
        return F.softmax(self.scaling_term * sim, dim=-1)

    def _make_patches(self, x_bfd):
        x = x_bfd.transpose(1, 2)
        x = F.pad(x, (self.win // 2, self.win // 2), mode="reflect")
        patches = x.unfold(dimension=2, size=self.win, step=1)
        return patches.permute(0, 2, 3, 1).contiguous()   # (B, F, W, d)

    def _agg_vel(self, seq_bfwd, level_w):
        return (seq_bfwd * level_w.view(1, 1, -1, 1)).sum(dim=2)

    def forward(self, mel_feature, gt_x_motion_f, gt_y_motion_f, gt_z_motion_f, inference):
        mel_feature = self.mel_encoder(mel_feature)

        mel_q = self.audio2query(mel_feature.transpose(1, 2)).transpose(1, 2)
        mel_addr = self.f2msim(feature=mel_q, memory=self.mel_memory)          # (B, F, S)

        r_seq_x = torch.einsum("bfs,swc->bfwc", mel_addr, self.motion_x_memory_seq)
        r_seq_y = torch.einsum("bfs,swc->bfwc", mel_addr, self.motion_y_memory_seq)
        r_seq_z = torch.einsum("bfs,swc->bfwc", mel_addr, self.motion_z_memory_seq)

        r_vel_x_f = self._agg_vel(r_seq_x, self.level_x_w)
        r_vel_y_f = self._agg_vel(r_seq_y, self.level_y_w)
        r_vel_z_f = self._agg_vel(r_seq_z, self.level_z_w)

        with torch.no_grad():
            gt_seq_vx = self._make_patches(gt_x_motion_f)
            gt_seq_vy = self._make_patches(gt_y_motion_f)
            gt_seq_vz = self._make_patches(gt_z_motion_f)

        aux_loss = \
            (1.0 - F.cosine_similarity(r_seq_x, gt_seq_vx, dim=-1)).mean() + \
            (1.0 - F.cosine_similarity(r_seq_y, gt_seq_vy, dim=-1)).mean() + \
            (1.0 - F.cosine_similarity(r_seq_z, gt_seq_vz, dim=-1)).mean()

        return self.drop(r_vel_x_f), self.drop(r_vel_y_f), self.drop(r_vel_z_f), aux_loss


# =============================================================================
# TAC: mesh-topology GCN + per-vertex TCN + region-adaptive gate   [stage 2 only]
# =============================================================================
def get_graph_structure(ply_path, device):
    """Normalized adjacency  D^-1/2 (A+I) D^-1/2  from mesh faces -> torch sparse COO (V,V)."""
    mesh = trimesh.load(ply_path, process=False)
    faces = mesh.faces.astype(np.int64)
    num_nodes = int(len(mesh.vertices))

    row = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    col = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.ones(row.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)

    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    adj = adj + sp.eye(num_nodes, dtype=np.float32, format="coo")

    degree = np.array(adj.sum(1)).reshape(-1).astype(np.float32)
    d_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt.astype(np.float32))
    adj_norm = (D_inv_sqrt @ adj @ D_inv_sqrt).tocoo()

    indices = torch.from_numpy(np.vstack([adj_norm.row, adj_norm.col]).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(adj_norm.shape)).coalesce().to(device)


class GraphConv(nn.Module):
    """out = Linear(A @ x);  x: (BT,V,Cin), adj: sparse (V,V)"""
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.linear = FC1(in_channels, out_channels)

    def forward(self, x, adj):
        BT, V, Cin = x.shape
        x_flat = x.transpose(0, 1).reshape(V, -1)          # (V, BT*Cin)
        ax_flat = torch.sparse.mm(adj, x_flat)
        ax = ax_flat.reshape(V, BT, Cin).transpose(0, 1)   # (BT, V, Cin)
        return self.linear(ax)


class GraphMotionIntegrator(nn.Module):
    """
    (GCN -> TCN) x 2 with a residual on the second block, then
        out = h_in + gate * delta_head(h),   gate = V * softmax(alpha * GATE_TAU) over vertices
    (alpha init 0 -> gate == 1 everywhere -> plain residual at init).
    """
    GATE_TAU = 16.0

    def __init__(self, args, ply_path):
        super().__init__()
        self.device = args.device
        self.hidden_dim = getattr(args, "gcn_hidden_dim", args.axis_mixer_dim)
        self.dropout_p = getattr(args, "gcn_dropout", 0.0)

        self.register_buffer("adj", get_graph_structure(ply_path, self.device))

        V = args.vertice_dim // 3
        H = self.hidden_dim
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(self.dropout_p)

        self.gcn1 = GraphConv(3, H)
        self.tcn1 = nn.Conv1d(H, H, kernel_size=3, padding=1)
        self.gcn2 = GraphConv(H, H)
        self.tcn2 = nn.Conv1d(H, H, kernel_size=3, padding=1)

        self.delta_head = nn.Linear(H, 3)
        self.alpha = nn.Parameter(torch.zeros(V))

    def _spatial_gcn(self, h_btvc, gcn_layer):
        B, T, V, C = h_btvc.shape
        h = gcn_layer(h_btvc.reshape(-1, V, C), self.adj)  # (BT,V,H); reshape: input is non-contiguous for B>1
        h = self.drop(self.act(h))
        return h.view(B, T, V, -1)

    def _temporal_tcn(self, h_btvh, tcn_layer):
        B, T, V, H = h_btvh.shape
        h_t = h_btvh.permute(0, 2, 3, 1).reshape(B * V, H, T)
        h_t = self.drop(self.act(tcn_layer(h_t)))
        return h_t.view(B, V, H, T).permute(0, 3, 1, 2)

    def forward(self, x_motion, y_motion, z_motion):
        B, T, V = x_motion.shape
        h_in = torch.stack([x_motion, y_motion, z_motion], dim=-1)          # (B,T,V,3)

        h = self._temporal_tcn(self._spatial_gcn(h_in, self.gcn1), self.tcn1)
        h = self._temporal_tcn(self._spatial_gcn(h, self.gcn2), self.tcn2) + h

        delta = self.delta_head(h)                                           # (B,T,V,3)
        gate = (F.softmax(self.alpha * self.GATE_TAU, dim=-1) * V).view(1, 1, V, 1)
        out = h_in + gate * delta
        return out[..., 0], out[..., 1], out[..., 2]


# =============================================================================
# SeeingSpeech
# =============================================================================
_DATASETS = ("vocaset", "TFHP")   # FLAME topology; the TAC graph is built from args.template_ply

LOSS_NAMES = {
    1: ["-", "L_vel", "L_dir", "L_store(1e-6)", "-", "-"],
    2: ["L_recon", "L_vel", "L_lap", "-", "-", "-"],
}


class SeeingSpeech(nn.Module):
    """
    args.stage == 1 : SAM                (returns per-direction losses)
    args.stage == 2 : SAM + TAC          (returns composed-motion losses)
    Submodule names are fixed: they are the state_dict keys of the released checkpoints.
    """
    def __init__(self, args):
        super().__init__()
        self.stage = int(getattr(args, "stage", 1))
        assert self.stage in (1, 2), f"stage must be 1 or 2, got {self.stage}"

        self.vertice_dim = args.vertice_dim
        self.mem_slot = args.mem_slot
        self.feature_dim = args.feature_dim
        self.device = args.device
        self.dataset = args.dataset
        self.mel_feature_dim = args.mel_feature_dim
        if self.dataset not in _DATASETS:
            raise ValueError(f"Unknown dataset in SeeingSpeech: {self.dataset}")
        self.VERTEX_WEIGHT = 1.0
        self.loss_names = LOSS_NAMES[self.stage]

        # z-score of the motion-encoder input: per vertex over time, as in script/main_autoencoder.py
        self.motion_norm_min_std = getattr(args, "motion_norm_min_std", 1e-5)

        # ---- content encoder (HuBERT, CNN feature encoder frozen) ----
        self.content_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.content_encoder.freeze_feature_encoder()
        self.content_mlp = FC3(in_dim=768, out_dim=self.feature_dim, hidden_dim=self.feature_dim)

        # ---- SAM ----
        self.mel_encoder = MelspecEncoder(mel_feature_dim=self.mel_feature_dim)
        self.PPE = PeriodicPositionalEncoding(self.feature_dim, period=args.period)
        self.x_transformer_decoder = self._make_decoder()
        self.y_transformer_decoder = self._make_decoder()
        self.z_transformer_decoder = self._make_decoder()
        self.vertice_x_map_r = FC1(self.feature_dim, self.vertice_dim // 3, is_encoder=False)
        self.vertice_y_map_r = FC1(self.feature_dim, self.vertice_dim // 3, is_encoder=False)
        self.vertice_z_map_r = FC1(self.feature_dim, self.vertice_dim // 3, is_encoder=False)
        self.memory_net = MemoryNetwork(args=args)

        # ---- motion encoder for L_store targets (copy of the motion-autoencoder encoder, frozen by the trainer) ----
        self.axis_vertex_encoder = AxisVertexTransformerEncoder(
            V=self.vertice_dim // 3, feature_dim=self.feature_dim,
            num_layers=1, num_heads=4, ff_mult=4.0, dropout=0.1,
            use_axis_embedding=getattr(args, "use_axis_embedding", False),
        )

        # ---- TAC (stage 2) ----
        if self.stage == 2:
            self.motion_integrator = GraphMotionIntegrator(args, args.template_ply)
            self.lap_loss_weight = getattr(args, "lap_loss_weight", 1e3)

    def _make_decoder(self):
        return nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.feature_dim, nhead=4, dim_feedforward=2 * self.feature_dim,
                batch_first=True, norm_first=True, activation=F.leaky_relu,
            ),
            num_layers=1,
        )

    # ------------------------------------------------------------------
    def _full_motion_zscore(self, x_btv_raw, min_std=1e-5):
        """(B,T,V): per-vertex z-score over time (std clamped to min_std)."""
        mean = x_btv_raw.mean(dim=1, keepdim=True)
        std = torch.clamp(x_btv_raw.std(dim=1, keepdim=True, unbiased=False), min=min_std)
        return (x_btv_raw - mean) / std

    def _laplacian(self, x_btv3):
        """L x = x - adj x  (adj = D^-1/2 (A+I) D^-1/2, reused from the integrator)."""
        B, T, V, C = x_btv3.shape
        flat = x_btv3.permute(2, 0, 1, 3).reshape(V, -1)
        sm = torch.sparse.mm(self.motion_integrator.adj, flat).reshape(V, B, T, C).permute(1, 2, 0, 3)
        return x_btv3 - sm

    # ------------------------------------------------------------------
    def forward(self, audio, template, vertice, one_hot, inference):
        """
        audio    : (B, L)          template : (B, 3V)        vertice : (B, T, 3V)   one_hot : unused
        returns  : (vertice_out (B,T,3V), slot1..slot6)  — see module docstring for slot meaning
        """
        zero = torch.zeros(1, device=self.device)
        B = template.shape[0]
        V = self.vertice_dim // 3
        template = template.unsqueeze(1)                                       # (B,1,3V)

        # ---- GT motion (raw) and its per-direction split ----
        gt_motion = (vertice.detach() - template.detach()) * self.VERTEX_WEIGHT
        gt_v3 = gt_motion.reshape(B, -1, V, 3)
        gt_x_motion, gt_y_motion, gt_z_motion = gt_v3[..., 0], gt_v3[..., 1], gt_v3[..., 2]

        # ---- L_store targets: frozen AxisAE encoder on z-scored GT motion ----
        x_in = self._full_motion_zscore(gt_x_motion, self.motion_norm_min_std)
        y_in = self._full_motion_zscore(gt_y_motion, self.motion_norm_min_std)
        z_in = self._full_motion_zscore(gt_z_motion, self.motion_norm_min_std)
        gt_x_motion_f, gt_y_motion_f, gt_z_motion_f = self.axis_vertex_encoder.encode_axes(x_in, y_in, z_in)

        # ---- audio features ----
        frame_num = vertice.shape[1]
        content_feature = self.content_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        mel_feature = linear_interpolation(self.mel_encoder(audio), frame_num)

        # ---- SAM: addressing + directional retrieval (+ L_store) ----
        r_motion_x_f, r_motion_y_f, r_motion_z_f, store_loss = self.memory_net(
            mel_feature=mel_feature,
            gt_x_motion_f=gt_x_motion_f, gt_y_motion_f=gt_y_motion_f, gt_z_motion_f=gt_z_motion_f,
            inference=inference,
        )

        # ---- SAM: content-guided refinement -> per-direction motion v̂_δ ----
        tgt_feature = self.PPE(self.content_mlp(content_feature))
        x_motion = self.vertice_x_map_r(self.x_transformer_decoder(tgt_feature, r_motion_x_f))   # (B,T,V)
        y_motion = self.vertice_y_map_r(self.y_transformer_decoder(tgt_feature, r_motion_y_f))
        z_motion = self.vertice_z_map_r(self.z_transformer_decoder(tgt_feature, r_motion_z_f))

        # ---- TAC: topology-aware composition (stage 2) ----
        if self.stage == 2:
            x_out, y_out, z_out = self.motion_integrator(x_motion, y_motion, z_motion)
        else:
            x_out, y_out, z_out = x_motion, y_motion, z_motion

        motion_out = torch.stack([x_out, y_out, z_out], dim=-1).flatten(2)     # (B,T,3V)
        vertice_out = motion_out / self.VERTEX_WEIGHT + template

        if inference:
            return vertice_out, zero, zero, zero, zero, zero, zero

        if self.stage == 1:
            return vertice_out, zero, *self._stage1_losses(
                x_motion, y_motion, z_motion, gt_x_motion, gt_y_motion, gt_z_motion, store_loss), zero, zero
        return vertice_out, *self._stage2_losses(vertice_out, vertice, motion_out, gt_motion), zero, zero, zero

    # ------------------------------------------------------------------
    @staticmethod
    def _stage1_losses(x_motion, y_motion, z_motion, gt_x, gt_y, gt_z, store_loss):
        """L_dir (per-direction MSE), L_vel (forward temporal difference MSE), 1e-6 * L_store."""
        gt_x, gt_y, gt_z = gt_x.detach(), gt_y.detach(), gt_z.detach()
        l_dir = F.mse_loss(x_motion, gt_x) + F.mse_loss(y_motion, gt_y) + F.mse_loss(z_motion, gt_z)
        l_vel = (
            F.mse_loss(x_motion[:, 1:] - x_motion[:, :-1], gt_x[:, 1:] - gt_x[:, :-1]) +
            F.mse_loss(y_motion[:, 1:] - y_motion[:, :-1], gt_y[:, 1:] - gt_y[:, :-1]) +
            F.mse_loss(z_motion[:, 1:] - z_motion[:, :-1], gt_z[:, 1:] - gt_z[:, :-1])
        )
        return l_vel, l_dir, 1e-6 * store_loss

    def _stage2_losses(self, vertice_out, vertice, motion_out, gt_motion):
        """L_recon (vertex MSE), L_vel (forward temporal difference MSE), λ_lap * L_lap  (Eq. 9)."""
        B = vertice_out.shape[0]
        V = self.vertice_dim // 3
        gt_motion = gt_motion.detach()
        pred_v3 = motion_out.reshape(B, -1, V, 3)
        gt_v3 = gt_motion.reshape(B, -1, V, 3)

        l_recon = F.mse_loss(vertice_out, vertice.detach())
        l_vel = F.mse_loss(motion_out[:, 1:] - motion_out[:, :-1], gt_motion[:, 1:] - gt_motion[:, :-1])
        l_lap = F.mse_loss(self._laplacian(pred_v3), self._laplacian(gt_v3)) * self.lap_loss_weight
        return l_recon, l_vel, l_lap
