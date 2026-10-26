#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Seeing Speech — motion autoencoder pre-training on VOCASET (script/main_autoencoder.py)
#   paper App. B.1 / Table S1: per direction, encoder = Linear -> PE -> 1 Transformer layer (latent z),
#   decoder = 1 Transformer layer -> Linear; D 128 / batch 1 / lr 1e-4 / 250 epochs; loss = L_recon + L_vel
# The encoder is shared by VOCASET and TFHP (both FLAME); its checkpoint is the default AE_CKPT of
# run_bash/run_vocaset.sh and run_bash/run_tfhp.sh.
#
# Usage:  bash run_bash/run_autoencoder.sh     (from any directory; paths below are relative to the repo root)
#   VOCASET_ROOT  data/vocaset
#   OUTPUT_ROOT   outputs        -> outputs/autoencoder/<SAVE_TAG>/axis_ae_best.pth
# Env overrides (all optional): VOCASET_ROOT / OUTPUT_ROOT / SAVE_TAG (vocaset_d128) / MAX_EPOCH (250)
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."     # repository root

VOCASET_ROOT=${VOCASET_ROOT:-data/vocaset}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
SAVE_TAG=${SAVE_TAG:-vocaset_d128}
MAX_EPOCH=${MAX_EPOCH:-250}

[[ -d "${VOCASET_ROOT}" ]] || { echo "[ERROR] VOCASET root not found: ${VOCASET_ROOT}" >&2; exit 1; }

python script/main_autoencoder.py --datasets vocaset --vocaset_root "${VOCASET_ROOT}" \
    --output_root "${OUTPUT_ROOT}" --save_tag "${SAVE_TAG}" --reset_save_dir \
    --feature_dim 128 --batch_size 1 --lr 1e-4 --max_epoch "${MAX_EPOCH}" \
    --num_layers 1 --lambda_velo 1.0

echo "[DONE] ${OUTPUT_ROOT}/autoencoder/${SAVE_TAG}/axis_ae_best.pth"
