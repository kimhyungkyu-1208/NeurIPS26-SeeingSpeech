#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeing Speech — visible articulatory distance and velocity errors (paper Sec. 4.3, Eq. 10, Table 2).

Region means of the lip anchors (assets/lip_anchor_{UL,LL,LMC,RMC}.npy, FLAME topology) at frame t:
    ICW(t) = |p_LMC,x(t) - p_RMC,x(t)|                                  inter-commissural width (spreading)
    ILD(t) = |p_UL,y(t)  - p_LL,y(t)|                                   inter-labial distance   (opening)
    LP(t)  = (p_UL,z(t) - p^neu_UL,z) + (p_LL,z(t) - p^neu_LL,z)        lip protrusion w.r.t. the neutral template
Distance error = RMSE between the GT and predicted signals over frames; velocity error = RMSE between their
first-order temporal differences. Both are computed per sequence and averaged over sequences.

Predictions: one <name>.npy (T, 5023*3) per test sequence, as written by script/main_2stage.py --test_only
(<seq>_condition_<subject>.npy is also accepted). GT is matched by name:
    vocaset : <gt_root>/<name>.npy           (60 fps, sub-sampled by --gt_stride 2 -> 30 fps)
    TFHP    : <gt_root>/<name>/wo_vert.npy   (25 fps)
Paths are relative to the working directory (run from the repository root).

  python evaluation/evaluate.py --dataset vocaset outputs/vocaset/result_<exp>
  python evaluation/evaluate.py --dataset tfhp    outputs/TFHP/result_<exp>
"""
import argparse
import os
import pickle

import numpy as np

DEFAULTS = {
    "vocaset": dict(gt_root="data/vocaset/vertices_npy", templates="data/vocaset/templates.pkl", gt_stride=2),
    "tfhp":    dict(gt_root="data/TFHP/TFHP_wo_Head_v03", templates="data/TFHP/TFHP_templates_v03.pkl", gt_stride=1),
}
MEASURES = ("ICW", "ILD", "LP")


# =========================
# I/O and name matching
# =========================
def load_templates_pkl(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def canonical_gt_filename(pred_file_name):
    """<seq>.npy | <seq>_condition_<subject>.npy | <seq>_pred_verts.npy -> <seq>.npy"""
    if "_condition_" in pred_file_name:
        return pred_file_name.split("_condition_")[0] + ".npy"
    if pred_file_name.endswith("_pred_verts.npy"):
        return pred_file_name[:-len("_pred_verts.npy")] + ".npy"
    return pred_file_name if pred_file_name.endswith(".npy") else pred_file_name + ".npy"


def extract_subject_id(file_name):
    """<subject>_<sentence or window>.npy -> <subject> (key of the templates pkl)"""
    name = os.path.basename(file_name)
    if name.endswith(".npy"):
        name = name[:-4]
    return name.rsplit("_", 1)[0] if "_" in name else name


def resolve_gt_path(gt_root, gt_name):
    flat = os.path.join(gt_root, gt_name)                                         # vocaset
    if os.path.exists(flat):
        return flat
    tfhp = os.path.join(gt_root, os.path.splitext(gt_name)[0], "wo_vert.npy")    # TFHP
    return tfhp if os.path.exists(tfhp) else flat


def to_index_1d(arr):
    a = np.asarray(arr)
    if a.dtype == bool:
        return np.where(a.ravel())[0].astype(np.int64)
    return a.astype(np.int64).ravel()


# =========================
# Metrics
# =========================
def region_mean(vertices_tv3, idx):
    return vertices_tv3[:, idx, :].mean(axis=1)   # (T, 3)


def articulatory_signals(vertices_abs_tv3, vertices_def_tv3, UL, LL, LMC, RMC):
    """Eq. 10. vertices_def_tv3 = vertices - neutral template (for LP)."""
    ULm, LLm = region_mean(vertices_abs_tv3, UL), region_mean(vertices_abs_tv3, LL)
    LMCm, RMCm = region_mean(vertices_abs_tv3, LMC), region_mean(vertices_abs_tv3, RMC)
    icw = np.abs(LMCm[:, 0] - RMCm[:, 0])
    ild = np.abs(ULm[:, 1] - LLm[:, 1])
    lp = region_mean(vertices_def_tv3, UL)[:, 2] + region_mean(vertices_def_tv3, LL)[:, 2]
    return {"ICW": icw, "ILD": ild, "LP": lp}


def _rmse_1d(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(a.shape[0], b.shape[0])
    if n < 1:
        return float("nan")
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def visible_articulatory_errors(g_abs, c_abs, g_def, c_def, UL, LL, LMC, RMC):
    """-> distance RMSE {ICW, ILD, LP} [m], velocity RMSE {ICW, ILD, LP} [m/frame]"""
    g = articulatory_signals(g_abs, g_def, UL, LL, LMC, RMC)
    p = articulatory_signals(c_abs, c_def, UL, LL, LMC, RMC)
    dist = {k: _rmse_1d(g[k], p[k]) for k in MEASURES}
    velo = {k: _rmse_1d(np.diff(g[k]), np.diff(p[k])) for k in MEASURES}
    return dist, velo


def evaluate_dir(pred_dir, gt_root, templates, gt_stride, UL, LL, LMC, RMC):
    dist = {k: [] for k in MEASURES}
    velo = {k: [] for k in MEASURES}
    n_total = n_used = n_miss_gt = n_miss_tpl = 0

    for pred_name in sorted(os.listdir(pred_dir)):
        if not pred_name.endswith(".npy"):
            continue
        n_total += 1

        gt_name = canonical_gt_filename(pred_name)
        gt_path = resolve_gt_path(gt_root, gt_name)
        if not os.path.exists(gt_path):
            n_miss_gt += 1
            continue

        g = np.load(gt_path)[::gt_stride]
        c = np.load(os.path.join(pred_dir, pred_name))
        T = min(g.shape[0], c.shape[0])
        if T < 3:
            continue

        tpl = templates.get(extract_subject_id(gt_name))
        if tpl is None:   # no template: the first GT frame is the reference for LP
            n_miss_tpl += 1
            tpl = g[0].reshape(-1, 3)
        else:
            tpl = np.asarray(tpl).reshape(-1, 3)

        g_abs = g[:T].reshape(T, -1, 3)
        c_abs = c[:T].reshape(T, -1, 3)
        d, v = visible_articulatory_errors(g_abs, c_abs, g_abs - tpl[None], c_abs - tpl[None], UL, LL, LMC, RMC)
        for k in MEASURES:
            dist[k].append(d[k])
            velo[k].append(v[k])
        n_used += 1

    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {
        "n_total": n_total, "n_used": n_used, "n_miss_gt": n_miss_gt, "n_miss_tpl": n_miss_tpl,
        "dist": {k: mean(dist[k]) for k in MEASURES},   # [m]
        "velo": {k: mean(velo[k]) for k in MEASURES},   # [m/frame]
    }


def main():
    p = argparse.ArgumentParser(description="Seeing Speech — visible articulatory distance / velocity errors")
    p.add_argument("pred_dirs", nargs="+", help="prediction directories (<name>.npy per test sequence)")
    p.add_argument("--dataset", choices=["vocaset", "tfhp"], default="vocaset", help="selects the default paths")
    p.add_argument("--gt_root", default=None, help="vocaset: data/vocaset/vertices_npy | tfhp: data/TFHP/TFHP_wo_Head_v03")
    p.add_argument("--templates", default=None, help="neutral templates pkl (LP reference)")
    p.add_argument("--gt_stride", type=int, default=None, help="GT frame stride (vocaset 60->30 fps: 2, tfhp: 1)")
    p.add_argument("--anchor_dir", default="assets", help="directory of lip_anchor_{UL,LL,LMC,RMC}.npy")
    args = p.parse_args()

    d = DEFAULTS[args.dataset]
    gt_root = args.gt_root or d["gt_root"]
    templates = load_templates_pkl(args.templates or d["templates"])
    gt_stride = args.gt_stride or d["gt_stride"]
    UL, LL, LMC, RMC = (to_index_1d(np.load(os.path.join(args.anchor_dir, f"lip_anchor_{n}.npy")))
                        for n in ("UL", "LL", "LMC", "RMC"))

    print(f"[INFO] dataset={args.dataset}  gt_root={gt_root}  gt_stride={gt_stride}")
    for pred_dir in args.pred_dirs:
        r = evaluate_dir(pred_dir, gt_root, templates, gt_stride, UL, LL, LMC, RMC)
        print(f"\n{pred_dir}")
        print(f"  files: {r['n_total']} total, {r['n_used']} used, {r['n_miss_gt']} without GT, "
              f"{r['n_miss_tpl']} without template")
        print("                      " + "  ".join(f"{k:>7s}" for k in MEASURES))
        print("  distance error [mm] " + "  ".join(f"{r['dist'][k] * 1e3:7.3f}" for k in MEASURES))
        print("  velocity error [mm] " + "  ".join(f"{r['velo'][k] * 1e3:7.3f}" for k in MEASURES) + "   (per frame)")


if __name__ == "__main__":
    main()
