# Seeing Speech: Learning Visible Articulatory Dynamics for Speech-Driven 3D Facial Animation

Hyung Kyu Kim, Byungchan Hwang, Hak Gu Kim — Chung-Ang University · NeurIPS 2026

**Project page:** https://irislab.cau.ac.kr/NeurIPS2026-SeeingSpeech/ · **Paper:** [OpenReview](https://openreview.net/forum?id=yLMGoDkvCM) · [arXiv](https://arxiv.org/abs/2609.30517)

Official training code. Seeing Speech models visible speech through three directional articulatory
motions (spreading, opening, protrusion = x, y, z): a Speech–Articulatory Memory (SAM) retrieves and
decodes the directional motions from speech, and a Topology-aware Articulatory Composition (TAC)
composes them over the mesh topology. Training has two stages (SAM, then TAC) on VOCASET or TFHP, after
pre-training a per-direction motion autoencoder that provides the memory-storage targets (`L_store`).
This repository contains code only: no datasets, no pretrained weights, no predictions.

## Files

| File | Role |
|---|---|
| `model/seeingspeech.py` | model definition (`SeeingSpeech`): SAM directional memories, decoders, TAC integrator, losses |
| `model/autoencoder.py` | per-direction motion autoencoder (`MotionAutoEncoder`) whose frozen encoder provides the memory-storage targets (`L_store`) |
| `model/hubert.py` | HuBERT speech encoder wrapper used by the model |
| `dataloader/loader.py` | VOCASET loader (audio `wav/`, `vertices_npy/`, `templates.pkl`, speaker splits); routes `--dataset TFHP` to the TFHP loader |
| `dataloader/tfhp_loader.py` | lazy, split-file based TFHP loader (100-frame windows) used by the main trainer and the autoencoder trainer |
| `script/main_autoencoder.py` | motion autoencoder pre-training |
| `script/main_1stage.py` | stage 1 entry point: SAM pre-training |
| `script/main_2stage.py` | stage 2 entry point: TAC fine-tuning, and inference (`--test_only`) |
| `script/trainer.py` | training / validation / test loop and command-line arguments shared by the two entry points (`--dataset vocaset` (default) or `TFHP`) |
| `run_bash/run_autoencoder.sh` | motion autoencoder pre-training on VOCASET |
| `run_bash/run_vocaset.sh` | VOCASET pipeline: stage 1 → stage 2 with the paper hyper-parameters |
| `run_bash/run_tfhp.sh` | TFHP pipeline: stage 1 → stage 2 with the paper hyper-parameters |
| `evaluation/evaluate.py` | visible articulatory distance / velocity errors (ICW, ILD, LP; paper Eq. 10, Table 2) |
| `assets/lips_coordinates.npy` | FLAME lip vertex indices used for the validation lip error (LVE) |
| `assets/lip_anchor_{UL,LL,LMC,RMC}.npy` | FLAME lip anchor vertex sets (25 / 23 / 33 / 33 vertices) used by `evaluation/evaluate.py` |

## Setup

```bash
pip install -r requirements.txt
```
Tested with Python 3.11, torch 2.5.1 (CUDA 12.1), torchaudio 2.5.1, transformers 4.51.3, numpy 2.1.2,
scipy 1.15.2, librosa 0.11.0, soundfile 0.13.1, trimesh 4.6.10. HuBERT (`facebook/hubert-base-ls960`) and the
Wav2Vec2 processor (`facebook/wav2vec2-base-960h`) are downloaded by transformers on first use.

Put the datasets under `data/` (symlinks are fine):

```
data/
├── vocaset/                          VOCASET
│   ├── wav/                          <seq>.wav
│   ├── vertices_npy/                 <seq>.npy   (T, 5023*3)
│   ├── templates.pkl                 {subject: (5023, 3) neutral mesh}
│   └── FLAME_sample.ply              FLAME mesh (topology of the TAC graph)
└── TFHP/                             TFHP, head-pose-free variant
    ├── train.txt, val.txt, test.txt  <person>/<seq> per line (speaker splits)
    ├── TFHP_wo_Head_v03/<person>_<seq>_<clip>/{audio.wav, wo_vert.npy (T, 5023, 3)}
    ├── TFHP_templates_v03.pkl        {<person>_<seq>: (5023, 3) neutral mesh}
    └── FLAME_sample.ply
```

## Run

The scripts can be called from any directory; they `cd` to the repository root, and every path is relative
to it. Outputs go to `outputs/`.

```bash
bash run_bash/run_autoencoder.sh   # 1. motion autoencoder -> outputs/autoencoder/vocaset_d128/axis_ae_best.pth
bash run_bash/run_vocaset.sh       # 2-3. VOCASET stage 1 -> stage 2   -> outputs/vocaset/{save,result}_<exp>
bash run_bash/run_tfhp.sh          # 2-3. TFHP stage 1 -> stage 2      -> outputs/TFHP/{save,result}_<exp>
```

Paths and hyper-parameters can be overridden with environment variables listed at the top of each script,
e.g. `DATASET_ROOT=/path/to/vocaset AE_CKPT=/path/to/axis_ae_best.pth bash run_bash/run_vocaset.sh`.
The device is set with `--device` (default `cuda`).

1. **Motion autoencoder pre-training**: `script/main_autoencoder.py` trains the per-direction motion
   autoencoder; its best checkpoint (`axis_ae_best.pth`) is passed to the main trainer as `--axis_ae_ckpt`
   and kept frozen. The VOCASET checkpoint is also used for TFHP (same FLAME topology).
2. **Stage 1 — SAM pre-training**: `script/main_1stage.py` learns the mel memory, directional value
   memories and per-direction decoders (memory window `--mem_win` = 2Ω+1, slots `--mem_slot`, feature
   dims `--feature_dim/--mel_feature_dim`, softmax temperature `--softmax_scaling_term`).
3. **Stage 2 — TAC fine-tuning**: `script/main_2stage.py --base_pretrained <stage-1 ckpt>` adds the motion
   integrator (hidden dim `--axis_mixer_dim`) and fine-tunes it with vertex reconstruction + velocity +
   Laplacian smoothness losses (`--velo_loss_weight`, `--lap_loss_weight`).
4. **Inference**: writes one prediction (`<seq>.npy`, (T, 5023*3)) per test sequence to
   `outputs/<dataset>/result_<exp>/`, using `outputs/<dataset>/save_<exp>/<N>_model.pth`:
   ```bash
   python script/main_2stage.py --test_only --exp_name <exp> --start_epoch <N> \
       --base_pretrained <stage-1 ckpt> [same model arguments as in run_bash/run_vocaset.sh]
   ```

After every epoch the model is validated on the val split; the best epoch is saved as
`<epoch>_model.pth` and used for the test-split predictions.

Both pipelines use the paper settings: d128 / md128 / 32 slots / window 2Ω+1 = 15, batch 1, LR 1e-4,
stage 1 250 epochs → stage 2 150 epochs, Laplacian weight 1e3; period 30 (`run_bash/run_vocaset.sh`) or 25
(`run_bash/run_tfhp.sh`).

## Evaluation

`evaluation/evaluate.py` computes the visible articulatory distance and velocity errors of the paper
(Eq. 10, Table 2): the inter-commissural width (ICW), the inter-labial distance (ILD) and the lip protrusion
relative to the neutral template (LP) are computed from the region means of the lip anchors for the
ground truth and the prediction; the distance error is their RMSE over frames, the velocity error the RMSE of
their frame-to-frame differences, averaged over the test sequences and reported in mm (per frame for the
velocity). It only needs numpy.

```bash
python evaluation/evaluate.py --dataset vocaset outputs/vocaset/result_<exp>   # GT: data/vocaset/vertices_npy (60 -> 30 fps)
python evaluation/evaluate.py --dataset tfhp    outputs/TFHP/result_<exp>      # GT: data/TFHP/TFHP_wo_Head_v03
```

## Not included

- The reconstruction metrics of the paper (FVE, LVE, FDD, LDTW).
- Datasets, pretrained weights, rendering.

## Third-party code

- `model/hubert.py` is adapted from the HuBERT implementation of HuggingFace Transformers
  (Apache License 2.0; the original header is kept, followed by a note on the modifications).
- The mel-spectrogram encoder in `model/seeingspeech.py` (`MelspecEncoder`, `Conv2d`) is adapted from the
  audio encoder of [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) (Prajwal et al., ACM MM 2020); see that
  repository for its license terms.

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). The third-party code listed
above keeps its own terms. The datasets (VOCASET, TFHP), the FLAME model and the pretrained HuBERT weights are
not distributed here and are subject to their own licenses and terms of use.

## Citation

```bibtex
@inproceedings{kim2026seeingspeech,
  title     = {Seeing Speech: Learning Visible Articulatory Dynamics for Speech-Driven 3D Facial Animation},
  author    = {Kim, Hyung Kyu and Hwang, Byungchan and Kim, Hak Gu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=yLMGoDkvCM}
}
```
