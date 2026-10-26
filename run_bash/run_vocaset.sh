#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Seeing Speech — VOCASET stage 1 (SAM, script/main_1stage.py) -> stage 2 (TAC, script/main_2stage.py)
# Paper settings (Sec. 4.2, App. B): Adam lr 1e-4, batch 1, stage 1 250 epochs, stage 2 150 epochs,
#   N 32 slots / Ω 7 (mem_win = 2Ω+1 = 15) / D_SAM 128 / D_TAC 64 / τ 16 / period ρ 30 (VOCASET 30 fps) /
#   L_vel weight 1 / λ_lap 1e3.
#
# Usage:  bash run_bash/run_vocaset.sh        (from any directory; paths below are relative to the repo root)
#   DATASET_ROOT  data/vocaset                                       wav/, vertices_npy/, templates.pkl, FLAME_sample.ply
#   AE_CKPT       outputs/autoencoder/vocaset_d128/axis_ae_best.pth  written by run_bash/run_autoencoder.sh
#   LIP_INDEX     assets/lips_coordinates.npy
#   OUTPUT_ROOT   outputs                                            -> outputs/vocaset/{save,result}_<exp>
# Env overrides (all optional): the four paths above /
#   STAGE1_EXP / STAGE2_EXP / STAGE1_TMP_EPOCH / STAGE1_CKPT / MAX_EPOCH_S1 / MAX_EPOCH_S2 / LR / LR_LOADED / LAP_W / VELO_W /
#   START_EPOCH_S1 / START_EPOCH_S2 / SKIP_STAGE1=1 / SKIP_STAGE2=1
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."     # repository root

DATASET_ROOT=${DATASET_ROOT:-data/vocaset}
AE_CKPT=${AE_CKPT:-outputs/autoencoder/vocaset_d128/axis_ae_best.pth}
LIP_INDEX=${LIP_INDEX:-assets/lips_coordinates.npy}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}

FEATURE_DIM=128; MEL_DIM=128; MEM_SLOT=32; AXIS_MIXER=64; SOFTMAX_TAU=16.0; PERIOD=30; MEM_WIN=15
MAX_EPOCH_S1=${MAX_EPOCH_S1:-250}; MAX_EPOCH_S2=${MAX_EPOCH_S2:-150}
LR=${LR:-1e-4}; LR_LOADED=${LR_LOADED:-0}; LAP_W=${LAP_W:-1e3}; VELO_W=${VELO_W:-1.0}

STAGE1_EXP=${STAGE1_EXP:-SeeingSpeech_VOCA_s1_d128_md128_s32_w15}
STAGE1_TMP_EPOCH=${STAGE1_TMP_EPOCH:-${MAX_EPOCH_S1}}
STAGE1_CKPT=${STAGE1_CKPT:-${OUTPUT_ROOT}/vocaset/save_${STAGE1_EXP}/tmp_${STAGE1_TMP_EPOCH}_model.pth}
STAGE2_EXP=${STAGE2_EXP:-SeeingSpeech_VOCA_s2_d128_md128_s32_w15_from_tmp${STAGE1_TMP_EPOCH}}
START_EPOCH_S1=${START_EPOCH_S1:-1}; START_EPOCH_S2=${START_EPOCH_S2:-1}

COMMON=(--dataset vocaset --dataset_root "${DATASET_ROOT}" --lip_index "${LIP_INDEX}" --output_root "${OUTPUT_ROOT}"
        --vertice_dim $((5023*3)) --identities_class 8 --period "${PERIOD}"
        --feature_dim "${FEATURE_DIM}" --mel_feature_dim "${MEL_DIM}" --mem_slot "${MEM_SLOT}" --mem_win "${MEM_WIN}"
        --axis_mixer_dim "${AXIS_MIXER}" --softmax_scaling_term "${SOFTMAX_TAU}"
        --axis_ae_ckpt "${AE_CKPT}")

[[ -d "${DATASET_ROOT}" ]] || { echo "[ERROR] dataset_root not found: ${DATASET_ROOT}" >&2; exit 1; }
[[ -f "${AE_CKPT}" ]] || { echo "[ERROR] AE ckpt not found: ${AE_CKPT} (run run_bash/run_autoencoder.sh)" >&2; exit 1; }

if [[ "${SKIP_STAGE1:-0}" != "1" ]]; then
    echo "================ [STAGE 1] SAM pre-training : ${STAGE1_EXP} ================"
    python script/main_1stage.py --exp_name "${STAGE1_EXP}" --start_epoch "${START_EPOCH_S1}" \
        --lr 1e-4 --max_epoch "${MAX_EPOCH_S1}" "${COMMON[@]}"
fi

[[ -f "${STAGE1_CKPT}" ]] || { echo "[ERROR] stage1 ckpt not found: ${STAGE1_CKPT}" >&2; exit 1; }

if [[ "${SKIP_STAGE2:-0}" != "1" ]]; then
    echo "================ [STAGE 2] TAC fine-tuning : ${STAGE2_EXP} ================"
    python script/main_2stage.py --exp_name "${STAGE2_EXP}" --start_epoch "${START_EPOCH_S2}" \
        --lr "${LR}" --lr_loaded "${LR_LOADED}" --weight_decay 0 \
        --lap_loss_weight "${LAP_W}" --velo_loss_weight "${VELO_W}" \
        --max_epoch "${MAX_EPOCH_S2}" --base_pretrained "${STAGE1_CKPT}" "${COMMON[@]}"
fi

echo "[DONE] stage1: ${OUTPUT_ROOT}/vocaset/save_${STAGE1_EXP}   stage2: ${OUTPUT_ROOT}/vocaset/save_${STAGE2_EXP}"
