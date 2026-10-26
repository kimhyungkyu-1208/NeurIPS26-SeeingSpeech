#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeing Speech — motion autoencoder (paper App. B.1, Table S1): one encoder–decoder pair per direction x, y, z.

  encoder : Linear(V -> D) -> sinusoidal PE -> Transformer layer       -> latent z (T, D)
  decoder : Transformer layer -> Linear(D -> V)
  Transformer layers: 4 heads, FFN 4*D, GELU, dropout 0.1. The decoder layer has no external memory to
  attend to, so it is a self-attention layer like the encoder's.

Pre-trained with script/main_autoencoder.py; the frozen `encoder` provides the reference motion features z of
L_store (its weights are copied into SeeingSpeech.axis_vertex_encoder by script/trainer.py).
"""

import math

import torch
import torch.nn as nn


# =========================
# Positional Encoding (time dimension)
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
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


# =========================
# Per-direction (x/y/z) Transformer encoder (time axis) - 3 Transformers
# =========================
class MotionEncoder(nn.Module):
    def __init__(
        self,
        V,
        feature_dim,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        ff_mult=4,
        use_axis_embedding=False,
    ):
        super(MotionEncoder, self).__init__()
        self.V = V
        self.feature_dim = feature_dim
        self.use_axis_embedding = bool(use_axis_embedding)

        # per-direction input projection: (V -> d)
        self.enc_x_in = nn.Linear(V, feature_dim)
        self.enc_y_in = nn.Linear(V, feature_dim)
        self.enc_z_in = nn.Linear(V, feature_dim)

        # shared sinusoidal positional encoding
        self.pos_encoding = PositionalEncoding(feature_dim, dropout=dropout)

        # one Transformer per direction (no parameter sharing)
        self.transformer_x = _build_transformer_encoder(feature_dim, num_heads, ff_mult, dropout, num_layers)
        self.transformer_y = _build_transformer_encoder(feature_dim, num_heads, ff_mult, dropout, num_layers)
        self.transformer_z = _build_transformer_encoder(feature_dim, num_heads, ff_mult, dropout, num_layers)

        # optional learnable direction embedding (3, D); off by default (paper Table S1 has none)
        # 0=X, 1=Y, 2=Z
        if self.use_axis_embedding:
            self.axis_embed = nn.Embedding(3, feature_dim)
            nn.init.normal_(self.axis_embed.weight, mean=0.0, std=0.02)
        else:
            self.axis_embed = None

        self._init_weights()

    def _init_weights(self):
        for m in [self.enc_x_in, self.enc_y_in, self.enc_z_in]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def _encode_axis(self, axis_in, enc_in, transformer, axis_id: int):
        """
        axis_in: (B, T, V)
        return: (B, T, d)
        """
        h = enc_in(axis_in)  # (B, T, d)

        if self.use_axis_embedding:
            e = self.axis_embed(torch.tensor(axis_id, device=h.device)).view(1, 1, -1)  # (1,1,d)
            h = h + e  # broadcast to (B,T,d)

        h = self.pos_encoding(h)  # (B, T, d)
        h = transformer(h)        # (B, T, d)
        return h

    def encode_axes(self, x_in, y_in, z_in):
        h_x = self._encode_axis(x_in, self.enc_x_in, self.transformer_x, axis_id=0)
        h_y = self._encode_axis(y_in, self.enc_y_in, self.transformer_y, axis_id=1)
        h_z = self._encode_axis(z_in, self.enc_z_in, self.transformer_z, axis_id=2)
        return h_x, h_y, h_z

    def get_latent_flat(self, x_in, y_in, z_in):
        h_x, h_y, h_z = self.encode_axes(x_in, y_in, z_in)
        B, T, D = h_x.shape
        z_x = h_x.reshape(B * T, D)
        z_y = h_y.reshape(B * T, D)
        z_z = h_z.reshape(B * T, D)
        return z_x, z_y, z_z


# =========================
# Per-direction (x/y/z) motion autoencoder
# =========================
class MotionAutoEncoder(nn.Module):
    def __init__(
        self,
        V,
        feature_dim,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        ff_mult=4,
        use_axis_embedding=False,
    ):
        super(MotionAutoEncoder, self).__init__()
        self.V = V
        self.feature_dim = feature_dim

        self.encoder = MotionEncoder(
            V=V,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ff_mult=ff_mult,
            use_axis_embedding=use_axis_embedding,
        )

        # per-direction decoder: Transformer layer(s) on z -> output projection (d -> V)
        self.dec_transformer_x = _build_transformer_encoder(feature_dim, num_heads, ff_mult, dropout, num_layers)
        self.dec_transformer_y = _build_transformer_encoder(feature_dim, num_heads, ff_mult, dropout, num_layers)
        self.dec_transformer_z = _build_transformer_encoder(feature_dim, num_heads, ff_mult, dropout, num_layers)
        self.dec_x = nn.Linear(feature_dim, V)
        self.dec_y = nn.Linear(feature_dim, V)
        self.dec_z = nn.Linear(feature_dim, V)

        self._init_weights()

    def _init_weights(self):
        for m in [self.dec_x, self.dec_y, self.dec_z]:
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x_in, y_in, z_in):
        h_x, h_y, h_z = self.encoder.encode_axes(x_in, y_in, z_in)  # latent z: (B,T,d)
        recon_x = self.dec_x(self.dec_transformer_x(h_x))  # (B,T,V)
        recon_y = self.dec_y(self.dec_transformer_y(h_y))
        recon_z = self.dec_z(self.dec_transformer_z(h_z))
        return recon_x, recon_y, recon_z

    def get_hidden(self, x_in, y_in, z_in):
        return self.encoder.encode_axes(x_in, y_in, z_in)

    def get_latent_flat(self, x_in, y_in, z_in):
        return self.encoder.get_latent_flat(x_in, y_in, z_in)
