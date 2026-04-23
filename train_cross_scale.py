"""
train_cross_scale.py
====================
Final production-ready cross-scale GBDT pipeline.

Critical Fixes Applied (v6.5 — Custom Metric Signature Fix)
-------------------------------------------------------------------
  1. [v6.5 FIX] Fixed LightGBM KeyError by removing the boolean 3rd return 
     value from custom metric functions (_f2_lgb, _f1_lgb). LightGBM 
     expects strictly (name_string, value_float), and 3-element tuples break it.
  2. [v6.4 FIX] Adjusted default threading for 7-core nodes: 
     --parallel_folds 3 and --gbdt_n_jobs 2 (6 cores total, 1 for OS).
  3. [v6.3 FIX] Replaced 3-hop bridge path counting with a strict 
     Easley & Kleinberg "Local Bridge" boolean (is_local_bridge). 
  4. [v6.3 FIX] Reduced feature count from 13/src to 12/src.
  5. [v6.2 FIX] Cleaned up redundant argparse boolean flags.
  6. [v6.1 FIX] Training curves plot F1 on right axis for interpretation.
  7. [v6 FIX] F2 Early Stopping, LR=0.1, Patience=150.
  8. [v5 FIX] Re-introduced Target-Scale features for ablation.
  9. [v5 FIX] Memory Evaluation Speedup: O(1) precomputed 2-hop lookups.
 10. Exact Zero-Overlap Lag: first_lag = (2 * target_w) // 5.
 11. F2-Aligned Thresholding & Native Class Imbalance Handling.
"""

import argparse
import csv
import gc
import io
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from matplotlib.patches import Patch

import parameters as p
from extract_distance_matrices import SP500_TICKERS_457, reorder_by_gics

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
except ImportError:
    LGBMClassifier = None
    lgb = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import shap
except ImportError:
    shap = None

from joblib import Parallel, delayed

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_W = 35
BAD_INDICES = [6, 111, 128, 169, 170, 225]
N_STOCKS = 457
WEEKLY_STRIDE = 5

EPSILON_CONFIGS = {35: 0.1, 70: 0.25, 120: 0.4, 180: 0.6}

DEFAULT_PKL_DIR = (
    "../Quasi_Differentiation_High_Temporal_Resolution_Cross_Correlations/Codes/"
    "Extract distance matrix (2017-2022) from pkl file"
)

def _min_safe_lag(target_w: int) -> int:
    assert target_w % WEEKLY_STRIDE == 0
    return (2 * target_w) // WEEKLY_STRIDE

def _get_scale_dependent_l1(target_w: int) -> float:
    return 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_adj_weekly(w: int, pkldir: str) -> Tuple[np.ndarray, List[str], List[str]]:
    epsilon = EPSILON_CONFIGS.get(w, 0.1)
    path = os.path.join(pkldir, f"IQDw{w}.pkl")
    if not os.path.isfile(path): raise FileNotFoundError(f"pkl not found: {path}")
    print(f"  Loading IQDw{w}.pkl ...", end=" ", flush=True)
    with open(path, "rb") as f: data = pickle.load(f).astype(np.float32)
    data = np.clip(data, -1.0, 1.0)
    dist = np.sqrt(2.0 * (1.0 - data)).astype(np.float32)
    adj = (dist <= epsilon).astype(np.float32)
    adj[:, np.arange(dist.shape[1]), np.arange(dist.shape[1])] = 0.0
    del data, dist
    adj = np.delete(adj, BAD_INDICES, axis=1); adj = np.delete(adj, BAD_INDICES, axis=2)
    assert adj.shape[1:] == (N_STOCKS, N_STOCKS)
    _, tickers_gics, sector_labels = reorder_by_gics(adj[0])
    ticker_to_pos = {t: i for i, t in enumerate(SP500_TICKERS_457)}
    perm_arr = np.array([ticker_to_pos[t] for t in tickers_gics], dtype=np.intp)
    adj = adj[:, perm_arr, :][:, :, perm_arr]
    adj_weekly = adj[::WEEKLY_STRIDE]
    print(f"done  weekly={adj_weekly.shape}")
    return adj_weekly.astype(np.float32), sector_labels, tickers_gics


# ─────────────────────────────────────────────────────────────────────────────
# Baselines & Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _marginal_prior_baseline(adj_target, train_idx, eval_idx):
    triu_i, triu_j = np.triu_indices(N_STOCKS, k=1)
    train_pos_rate = float(adj_target[train_idx][:, triu_i, triu_j].mean())
    y_true = adj_target[eval_idx][:, triu_i, triu_j].ravel()
    return y_true, np.full(len(y_true), train_pos_rate, dtype=np.float32)

def _short_scale_oracle_baseline(adj_source, adj_target, eval_idx, first_lag):
    assert (eval_idx >= first_lag).all()
    triu_i, triu_j = np.triu_indices(N_STOCKS, k=1)
    y_true = adj_target[eval_idx][:, triu_i, triu_j].ravel()
    y_score = adj_source[eval_idx - first_lag][:, triu_i, triu_j].ravel()
    return y_true, y_score

def _compute_metrics(y_true, y_score, threshold=0.5, name=""):
    y_pred = (y_score >= threshold).astype(np.int32)
    n_pos = y_true.sum()
    if n_pos == 0 or n_pos == len(y_true): auc = ap = float("nan")
    else: auc, ap = roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)
    f1, prec, rec = f1_score(y_true, y_pred, zero_division=0), precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_score)
    f2 = float((5.0 * prec * rec) / (4.0 * prec + rec + 1e-12))
    print(f"      {name:<42s}  AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  F2={f2:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Brier={brier:.6f}")
    return dict(name=name, auc=auc, ap=ap, f1=f1, f2=f2, prec=prec, rec=rec, brier=brier)

def _compute_metrics_silent(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(np.int32)
    n_pos = y_true.sum()
    if n_pos == 0 or n_pos == len(y_true): auc = ap = float("nan")
    else: auc, ap = roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f2 = float((5.0 * prec * rec) / (4.0 * prec + rec + 1e-12))
    return dict(auc=auc, ap=ap, f1=f1_score(y_true, y_pred, zero_division=0), f2=f2, prec=prec, rec=rec, brier=brier_score_loss(y_true, y_score))

def _best_f2_threshold(y_true, y_score):
    if y_true.sum() == 0 or y_true.sum() == len(y_true): return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f2_vals = (5.0 * precision[:-1] * recall[:-1]) / (4.0 * precision[:-1] + recall[:-1] + 1e-12)
    if len(f2_vals) == 0: return 0.5, 0.0
    best_idx = int(np.argmax(f2_vals))
    return float(thresholds[best_idx]), float(f2_vals[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def _fit_calibrator(y_true_calib, y_score_calib, method):
    y_true_calib = y_true_calib.astype(np.int32); y_score_calib = np.clip(y_score_calib.astype(np.float64), 1e-7, 1.0 - 1e-7)
    n_pos = int(y_true_calib.sum())
    if n_pos == 0 or n_pos == len(y_true_calib) or method == "none": return None, "none"
    candidates = {}
    if method in ("auto", "platt"):
        logits = np.log(y_score_calib / (1.0 - y_score_calib)).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=0); lr.fit(logits, y_true_calib)
        candidates["platt"] = (lr, float(brier_score_loss(y_true_calib, lr.predict_proba(logits)[:, 1])))
    if method in ("auto", "isotonic"):
        iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(y_score_calib, y_true_calib)
        candidates["isotonic"] = (iso, float(brier_score_loss(y_true_calib, iso.predict(y_score_calib))))
    if not candidates: return None, "none"
    chosen = min(candidates, key=lambda k: candidates[k][1]) if method == "auto" else method
    return candidates[chosen][0], chosen

def _apply_calibrator(calibrator, method, y_score_raw):
    if calibrator is None or method == "none": return y_score_raw.astype(np.float32)
    y_score_raw = np.clip(y_score_raw.astype(np.float64), 1e-7, 1.0 - 1e-7)
    if method == "platt":
        logits = np.log(y_score_raw / (1.0 - y_score_raw)).reshape(-1, 1); return calibrator.predict_proba(logits)[:, 1].astype(np.float32)
    return calibrator.predict(y_score_raw).astype(np.float32)

def _calibrate_on_calib_eval_on_test(y_true_calib, y_score_calib_raw, y_true_test, y_score_test_raw, method):
    brier_raw_test = float(brier_score_loss(y_true_test, y_score_test_raw))
    calibrator, chosen_method = _fit_calibrator(y_true_calib, y_score_calib_raw, method)
    y_test_cal = _apply_calibrator(calibrator, chosen_method, y_score_test_raw)
    brier_cal_test = float(brier_score_loss(y_true_test, y_test_cal))
    info = {"brier_raw": brier_raw_test, "brier_calibrated": brier_cal_test, "brier_platt": float("nan"), "brier_isotonic": float("nan")}
    for m in ("platt", "isotonic"):
        if method in ("auto", m):
            c, _ = _fit_calibrator(y_true_calib, y_score_calib_raw, m)
            if c is not None: info[f"brier_{m}"] = float(brier_score_loss(y_true_test, _apply_calibrator(c, m, y_score_test_raw)))
    return y_test_cal, chosen_method, info


# ─────────────────────────────────────────────────────────────────────────────
# Feature Construction (v6.3: True Local Bridge Boolean)
# ─────────────────────────────────────────────────────────────────────────────

def _feature_names(source_ws: List[int], target_w: int, history_lags: int, first_lag: int) -> List[str]:
    names = ["same_sector"]
    for k in range(first_lag, first_lag + history_lags):
        names.extend([f"w{target_w}_edge_lag{k}", f"deg{target_w}_i_lag{k}", f"deg{target_w}_j_lag{k}", f"deg{target_w}_absdiff_lag{k}", f"common_nbrs_w{target_w}_lag{k}", f"jaccard_w{target_w}_lag{k}"])
        for ws in source_ws:
            names.extend([f"w{ws}_edge_lag{k}", f"is_local_bridge_w{ws}_lag{k}", f"deg{ws}_i_lag{k}", f"deg{ws}_j_lag{k}", f"deg{ws}_absdiff_lag{k}", f"common_nbrs_w{ws}_lag{k}", f"jaccard_w{ws}_lag{k}"])
            names.extend([f"neckness_w{ws}_i_lag{k}", f"neckness_w{ws}_j_lag{k}", f"cross_sector_deg_w{ws}_i_lag{k}", f"cross_sector_deg_w{ws}_j_lag{k}", f"clust_boundary_diff_w{ws}_lag{k}"])
    return names

def _get_ablation_mask(feature_names: List[str], ablation: str, target_w: int) -> np.ndarray:
    if ablation == "none": return np.ones(len(feature_names), dtype=bool)
    elif ablation == "pure_cross_scale":
        return np.array([not (n.startswith(f"w{target_w}_") or n.startswith(f"deg{target_w}_") or n.startswith(f"common_nbrs_w{target_w}_") or n.startswith(f"jaccard_w{target_w}_")) for n in feature_names])
    raise ValueError(f"Unknown ablation: {ablation!r}")

def _build_pair_features_for_t(t, pair_i, pair_j, adj_sources, adj_target, sector_ids, history_lags, first_lag, target_w):
    feats = [(sector_ids[pair_i] == sector_ids[pair_j]).astype(np.float32)]
    same_sec_mat = (sector_ids[:, None] == sector_ids[None, :]).astype(np.float32)
    
    for k in range(first_lag, first_lag + history_lags):
        step = max(t - k, 0)
        
        # Target-Scale
        tgt = adj_target[step]; tgt_deg = tgt.sum(axis=1); tgt_s2 = tgt @ tgt
        tgt_com = tgt_s2[pair_i, pair_j]; tgt_uni = np.clip(tgt_deg[pair_i] + tgt_deg[pair_j] - tgt_com, 1, None)
        feats.extend([tgt[pair_i, pair_j], tgt_deg[pair_i], tgt_deg[pair_j], np.abs(tgt_deg[pair_i] - tgt_deg[pair_j]), tgt_com, tgt_com / tgt_uni])
        
        # Source-Scale + Neck
        for ws, adj_s in adj_sources.items():
            s = adj_s[step]; deg_s = s.sum(axis=1); s2 = s @ s
            
            # O(1) Lookups
            com_s = s2[pair_i, pair_j]; uni_s = np.clip(deg_s[pair_i] + deg_s[pair_j] - com_s, 1, None)
            
            # True Easley & Kleinberg Local Bridge boolean (Array-safe)
            is_bridge = ((s[pair_i, pair_j] > 0) & (com_s == 0)).astype(np.float32)
            
            feats.extend([
                s[pair_i, pair_j],          
                is_bridge,                
                deg_s[pair_i],             
                deg_s[pair_j],             
                np.abs(deg_s[pair_i] - deg_s[pair_j]),  
                com_s,                     
                com_s / uni_s              
            ])
            
            # Topological Features
            triangles_i = (s * s2).sum(axis=1) / 2.0
            clust_coeff = (2.0 * triangles_i) / (deg_s * (deg_s - 1.0) + 1e-6)
            neckness = 1.0 - clust_coeff
            cross_sec_adj = s * (1.0 - same_sec_mat)
            cross_deg = cross_sec_adj.sum(axis=1)
            
            feats.extend([
                neckness[pair_i],                                         
                neckness[pair_j],                                         
                cross_deg[pair_i],                                        
                cross_deg[pair_j],                                        
                np.abs(clust_coeff[pair_i] - clust_coeff[pair_j])        
            ])
            
    return np.column_stack(feats).astype(np.float32)

def _sample_pairs_from_target(adj_target, neg_ratio, rng):
    triu_i, triu_j = np.triu_indices(N_STOCKS, k=1); y = adj_target[triu_i, triu_j].astype(np.int32)
    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos_idx) == 0: chosen = rng.choice(neg_idx, size=min(len(neg_idx), 4096), replace=False)
    else:
        n_neg = min(len(neg_idx), neg_ratio * len(pos_idx))
        neg_chosen = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=np.int64)
        chosen = np.concatenate([pos_idx, neg_chosen]); rng.shuffle(chosen)
    return np.stack([triu_i[chosen], triu_j[chosen]], axis=1), y[chosen]

def _build_train_matrix(train_idx, adj_sources, adj_target, sector_ids, history_lags, first_lag, target_w, neg_ratio, rng):
    x_parts, y_parts = [], []
    for t in train_idx:
        pairs, labels = _sample_pairs_from_target(adj_target[int(t)], neg_ratio=neg_ratio, rng=rng)
        if len(pairs) == 0: continue
        x_parts.append(_build_pair_features_for_t(int(t), pairs[:, 0], pairs[:, 1], adj_sources, adj_target, sector_ids, history_lags, first_lag, target_w))
        y_parts.append(labels.astype(np.int32))
    if not x_parts:
        n_feats = 1 + (12 * len(adj_sources) + 6) * history_lags
        return np.zeros((0, n_feats), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.concatenate(x_parts), np.concatenate(y_parts)

def _build_eval_matrix(eval_idx, adj_sources, adj_target, sector_ids, history_lags, first_lag, target_w):
    triu_i, triu_j = np.triu_indices(N_STOCKS, k=1); x_parts, y_parts = [], []
    for t in eval_idx:
        x_parts.append(_build_pair_features_for_t(int(t), triu_i, triu_j, adj_sources, adj_target, sector_ids, history_lags, first_lag, target_w))
        y_parts.append(adj_target[int(t)][triu_i, triu_j].astype(np.int32))
    return np.concatenate(x_parts), np.concatenate(y_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_dataset_timeline(all_target_idx, folds, t_w, first_lag, out_dir, prefix):
    if plt is None: return
    n_folds = len(folds); fig, axes = plt.subplots(n_folds, 1, figsize=(14, 2.5 * n_folds), sharex=True)
    if n_folds == 1: axes = [axes]
    colors = ["steelblue", "lightblue", "lightgray", "orange", "red"]; labels = ["Train (Rolling)", "Train (Tail-Eval)", "Gap", "Calibration", "Test"]
    for i, (ax, (tr_idx, ca_idx, te_idx)) in enumerate(zip(axes, folds)):
        tail_n = max(10, int(len(tr_idx) * 0.2)); tr_early_end = tr_idx[-tail_n-1] if tail_n < len(tr_idx) else tr_idx[0]
        ax.axvspan(tr_idx[0], tr_early_end, color=colors[0], alpha=0.6); ax.axvspan(tr_idx[-tail_n], tr_idx[-1], color=colors[1], alpha=0.8)
        ax.axvspan(tr_idx[-1]+1, ca_idx[0]-1, color=colors[2], alpha=0.4); ax.axvspan(ca_idx[0], ca_idx[-1], color=colors[3], alpha=0.6); ax.axvspan(te_idx[0], te_idx[-1], color=colors[4], alpha=0.6)
        ax.set_yticks([]); ax.set_ylabel(f"Fold {i+1}", fontsize=10, fontweight='bold', rotation=0, labelpad=25)
        ax.text(0.98, 0.5, f"Tr:{len(tr_idx)} Ca:{len(ca_idx)} Te:{len(te_idx)}", transform=ax.transAxes, ha='right', va='center', fontsize=8, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    legend_elements = [Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    axes[0].legend(handles=legend_elements, loc="upper left", ncol=5, fontsize=8); axes[-1].set_xlim(0, t_w); axes[-1].set_xlabel("Weekly Snapshot Index (t)")
    fig.suptitle(f"Temporal Data Split Overview (first_lag={first_lag} weeks)", fontsize=12, y=1.01); fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_data_timeline.png", dpi=150, bbox_inches='tight'); plt.close(fig)

def _reorder_features_by_category(feature_names, target_w):
    cat_order = ["Sector Prior", "Target-Scale Attrs", "Source-Scale Standard Attrs", "Source-Scale Topological (Neck/Boundary)"]
    mapping = {k: [] for k in cat_order}
    for i, name in enumerate(feature_names):
        if "same_sector" in name: mapping["Sector Prior"].append(i)
        elif (name.startswith(f"w{target_w}_edge") or name.startswith(f"deg{target_w}_") or name.startswith(f"common_nbrs_w{target_w}_") or name.startswith(f"jaccard_w{target_w}_")): mapping["Target-Scale Attrs"].append(i)
        elif "neckness" in name or "cross_sector_deg" in name or "clust_boundary" in name or "is_local_bridge" in name: mapping["Source-Scale Topological (Neck/Boundary)"].append(i)
        else: mapping["Source-Scale Standard Attrs"].append(i)
    ordered_indices, ordered_names, boundaries = [], [], [], []
    for cat in cat_order:
        if mapping[cat]: ordered_indices.extend(mapping[cat]); ordered_names.extend([feature_names[idx] for idx in mapping[cat]]); boundaries.append(len(ordered_names) - 0.5)
    return np.array(ordered_indices, dtype=np.intp), ordered_names, boundaries

def _get_snapshot_indices(y_sample, rng, n_per_class=10):
    pos_idx, neg_idx = np.where(y_sample == 1)[0], np.where(y_sample == 0)[0]
    if len(pos_idx) > 0 and len(neg_idx) > 0: 
        chosen_idx = np.concatenate([rng.choice(pos_idx, size=min(n_per_class, len(pos_idx)), replace=False), rng.choice(neg_idx, size=min(n_per_class, len(neg_idx)), replace=False)])
    else: chosen_idx = np.arange(min(20, len(y_sample)))
    rng.shuffle(chosen_idx); return chosen_idx

def _plot_feature_snapshot(x_sample, y_sample, feature_names, out_dir, prefix, tickers=None, chosen_idx=None, target_w=0):
    if plt is None: return
    if chosen_idx is None: chosen_idx = _get_snapshot_indices(y_sample, np.random.default_rng(0))
    ordered_idx, ordered_names, boundaries = _reorder_features_by_category(feature_names, target_w)
    sort_order = np.argsort(y_sample[chosen_idx]); sorted_idx = chosen_idx[sort_order]
    data = x_sample[sorted_idx][:, ordered_idx]; y_chosen = y_sample[sorted_idx]; n_rows = len(data)
    fig = plt.figure(figsize=(max(16, len(ordered_names)*0.25), n_rows*0.45)); ax = fig.add_axes([0.0, 0.0, 0.75, 1.0]); cax = ax.imshow(data, aspect='auto', cmap='viridis')
    if tickers is not None:
        triu_i, triu_j = np.triu_indices(N_STOCKS, k=1); n_pairs = len(triu_i)
        ytick_labels = [f"{tickers[triu_i[p % n_pairs]]}-{tickers[triu_j[p % n_pairs]]}" for p in sorted_idx]
    else: ytick_labels = [f"Pair {i}" for i in range(n_rows)]
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(ytick_labels, fontsize=7); ax.set_xticks(range(len(ordered_names))); ax.set_xticklabels(ordered_names, rotation=90, ha='right', fontsize=5)
    for b in boundaries[:-1]: ax.axvline(x=b, color='white', linewidth=2.0)
    fig.colorbar(cax, ax=ax, location='right', shrink=1.0, pad=0.05)
    label_x = len(ordered_names) + 0.2
    for i in range(n_rows): ax.text(label_x, i, f"y={int(y_chosen[i])}", va='center', ha='left', color='red' if y_chosen[i]==1 else 'black', fontsize=8, fontweight='bold', clip_on=False)
    ax.set_title("Feature Matrix Snapshot (Raw Values) - Grouped by y-value", fontsize=10); fig.savefig(out_dir / f"{prefix}_feature_snapshot.png", dpi=150, bbox_inches='tight'); plt.close(fig)

def _plot_feature_snapshot_standardized(x_sample, y_sample, feature_names, out_dir, prefix, tickers=None, chosen_idx=None, target_w=0):
    if plt is None: return
    if chosen_idx is None: chosen_idx = _get_snapshot_indices(y_sample, np.random.default_rng(0))
    ordered_idx, ordered_names, boundaries = _reorder_features_by_category(feature_names, target_w)
    sort_order = np.argsort(y_sample[chosen_idx]); sorted_idx = chosen_idx[sort_order]; n_rows = len(sorted_idx); y_chosen = y_sample[sorted_idx]
    vis_data = (x_sample[sorted_idx][:, ordered_idx] - x_sample[sorted_idx][:, ordered_idx].mean(axis=0)) / (x_sample[sorted_idx][:, ordered_idx].std(axis=0) + 1e-6)
    fig = plt.figure(figsize=(max(16, len(ordered_names)*0.25), n_rows*0.45)); ax = fig.add_axes([0.0, 0.0, 0.75, 1.0]); cax = ax.imshow(vis_data, aspect='auto', cmap='viridis', vmin=-2, vmax=2)
    if tickers is not None:
        triu_i, triu_j = np.triu_indices(N_STOCKS, k=1); n_pairs = len(triu_i)
        ytick_labels = [f"{tickers[triu_i[p % n_pairs]]}-{tickers[triu_j[p % n_pairs]]}" for p in sorted_idx]
    else: ytick_labels = [f"Pair {i}" for i in range(n_rows)]
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(ytick_labels, fontsize=7); ax.set_xticks(range(len(ordered_names))); ax.set_xticklabels(ordered_names, rotation=90, ha='right', fontsize=5)
    for b in boundaries[:-1]: ax.axvline(x=b, color='white', linewidth=2.0)
    fig.colorbar(cax, ax=ax, location='right', shrink=1.0, pad=0.05)
    label_x = len(ordered_names) + 0.2
    for i in range(n_rows): ax.text(label_x, i, f"y={int(y_chosen[i])}", va='center', ha='left', color='red' if y_chosen[i]==1 else 'black', fontsize=8, fontweight='bold', clip_on=False)
    ax.set_title("Feature Matrix Snapshot (Z-Score Standardized) - Grouped by y-value", fontsize=10); fig.savefig(out_dir / f"{prefix}_feature_snapshot_standardized.png", dpi=150, bbox_inches='tight'); plt.close(fig)

def _plot_baseline_pr_comparison(y_true, scores_dict, out_dir, prefix):
    if plt is None: return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors, linestyles = {"Marginal Prior": "gray", "Oracle": "black", "Full": "steelblue", "Pure Cross-Scale": "crimson"}, {"Marginal Prior": ":", "Oracle": "--", "Full": "-", "Pure Cross-Scale": "-."}
    base_rate = float(y_true.mean())
    for name, y_score in scores_dict.items():
        if name == "Marginal Prior": ax.plot([0.0, 1.0], [base_rate, base_rate], label=f"{name} (AP={base_rate:.4f})", color=colors.get(name, "blue"), linestyle=linestyles.get(name, "-"), linewidth=2.5); continue
        try: ap = average_precision_score(y_true, y_score)
        except ValueError: ap = float("nan")
        sort_idx = np.argsort(-y_score); sorted_scores, sorted_true = y_score[sort_idx], y_true[sort_idx]
        cum_tp = np.cumsum(sorted_true); cum_total = np.arange(1, len(sorted_true) + 1)
        precision, recall = cum_tp / cum_total, cum_tp / max(cum_tp[-1], 1)
        recall, precision = np.concatenate([[0.0], recall]), np.concatenate([[base_rate], precision])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})", color=colors.get(name, "blue"), linestyle=linestyles.get(name, "-"), linewidth=2 if "Prior" not in name else 1.5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall: Models vs Null Baselines (Final Test Set)"); ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_baseline_comparison_pr.png", dpi=150); plt.close(fig)

def _plot_training_curves(curves, out_dir, prefix):
    if plt is None or not curves: return
    palette = {"Full": "steelblue", "pure_cross_scale": "crimson"}; fig, ax1 = plt.subplots(figsize=(10, 6)); ax2 = ax1.twinx()
    for label, curve, best_iter in curves:
        c = palette.get(label, "gray"); ax1.plot(curve["iterations"], curve["f2"], label=f"{label} Max-F2 (stop)", color=c, linewidth=2); ax1.axvline(best_iter, color=c, linestyle="--", alpha=0.4)
        if "f1" in curve: ax2.plot(curve["iterations"], curve["f1"], label=f"{label} Max-F1 (ref)", color=c, linewidth=2, linestyle=":", alpha=0.8)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Max-F2 (Stopping Metric)", color="black"); ax2.set_ylabel("Max-F1 (Reference)", color="gray")
    ax1.set_title("Learning Curves: F2 (Left, Stopping) vs F1 (Right, Reference)"); ax1.grid(True, alpha=0.3)
    (l1, lb1), (l2, lb2) = ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels(); ax1.legend(l1+l2, lb1+lb2, fontsize=9, loc="center right"); fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_training_curve.png", dpi=150); plt.close(fig)

def _plot_model_diagnostics(y_true, y_score, threshold, out_dir, prefix):
    if plt is None: return
    pr, rc, th = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6,5)); ax.plot(rc, pr); ax.set_title("PR Curve (Final Test Set)"); fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_pr_curve.png", dpi=150); plt.close(fig)
    f1 = (2*pr[:-1]*rc[:-1])/(pr[:-1]+rc[:-1]+1e-12); f2 = (5*pr[:-1]*rc[:-1])/(4*pr[:-1]+rc[:-1]+1e-12)
    fig, ax = plt.subplots(figsize=(7,5)); ax.plot(th, f1, label="F1"); ax.plot(th, f2, label="F2"); ax.plot(th, pr[:-1], label="Prec"); ax.plot(th, rc[:-1], label="Rec")
    ax.axvline(threshold, linestyle="--", color="black", label=f"thr={threshold:.3f}"); ax.legend(); ax.set_title("Metrics vs Threshold (Final Test Set)"); fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_threshold_curves.png", dpi=150); plt.close(fig)
    cm = confusion_matrix(y_true, (y_score>=threshold).astype(int), labels=[0,1])
    fig, ax = plt.subplots(figsize=(5,4)); im=ax.imshow(cm, cmap="Blues"); ax.set_xticks([0,1]); ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticks([0,1]); ax.set_yticklabels(["True 0","True 1"])
    for r in range(2):
        for c in range(2): ax.text(c, r, str(cm[r,c]), ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix (Final Test Set)"); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=150); plt.close(fig)
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(6,5)); ax.plot(mean_pred, frac_pos, "o-", label="Model"); ax.plot([0,1],[0,1], "--", label="Perfect"); ax.legend(); ax.set_title("Calibration Curve (Final Test Set)"); fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_calibration_curve.png", dpi=150); plt.close(fig)
    except Exception: pass
    fig, ax = plt.subplots(figsize=(7,4)); ax.hist(y_score[y_true==0], bins=40, alpha=0.6, label="y=0"); ax.hist(y_score[y_true==1], bins=40, alpha=0.6, label="y=1")
    ax.axvline(threshold, linestyle="--", color="black", label=f"thr={threshold:.3f}"); ax.set_xlabel("Predicted Prob"); ax.set_ylabel("Count"); ax.set_title("Score Distribution by Class (Final Test Set)"); ax.legend(); fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_score_histogram.png", dpi=150); plt.close(fig)

def _plot_feature_importance(model, feature_names, out_dir, prefix):
    if plt is None or not hasattr(model, "feature_importances_"): return
    imp = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(imp) != len(feature_names): return
    idx = np.argsort(imp)[::-1][:20]; fig, ax = plt.subplots(figsize=(10,6)); ax.barh(np.array(feature_names)[idx][::-1], imp[idx][::-1]); ax.set_title("Top-20 Feature Importance"); fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_feature_importance.png", dpi=150); plt.close(fig)

def _plot_shap_summary(model, x_sample, feature_names, out_dir, prefix):
    if plt is None or shap is None: return
    try:
        explainer = shap.TreeExplainer(model); vals = explainer.shap_values(x_sample); v = vals[1] if isinstance(vals, list) and len(vals)==2 else vals
        fig = plt.figure(); shap.summary_plot(v, x_sample, feature_names=feature_names, show=False); plt.tight_layout(); fig.savefig(out_dir / f"{prefix}_shap_summary.png", dpi=150, bbox_inches='tight'); plt.close(fig)
    except Exception as err: print(f"  SHAP skipped: {err}")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Metrics & Model Fitting
# ─────────────────────────────────────────────────────────────────────────────

_f2_subsample_idx, _f2_subsample_n = None, -1
_f1_subsample_idx, _f1_subsample_n = None, -1

def _f2_lgb(y_true, y_pred):
    """Raw max-F2 score — used as primary early stopping metric."""
    global _f2_subsample_idx, _f2_subsample_n
    try:
        n = len(y_true); SUBSAMPLE_CAP = 50_000
        if n > SUBSAMPLE_CAP:
            if _f2_subsample_n != n: _f2_subsample_idx = np.random.default_rng(0).choice(n, size=SUBSAMPLE_CAP, replace=False); _f2_subsample_n = n
            yt, yp = y_true[_f2_subsample_idx], y_pred[_f2_subsample_idx]
        else: yt, yp = y_true, y_pred
        precision, recall, thresholds = precision_recall_curve(yt, yp)
        f2_vals = (5.0 * precision[:-1] * recall[:-1]) / (4.0 * precision[:-1] + recall[:-1] + 1e-12)
        raw_f2 = float(np.max(f2_vals)) if len(f2_vals) > 0 else 0.0
        # v6.5 FIX: Strictly return (name, value) tuple for LightGBM custom metrics
        return "f2", raw_f2
    except ValueError:
        return "f2", 0.0

def _f1_lgb(y_true, y_pred):
    """Raw max-F1 score — used as a visual reference metric."""
    global _f1_subsample_idx, _f1_subsample_n
    try:
        n = len(y_true); SUBSAMPLE_CAP = 50_000
        if n > SUBSAMPLE_CAP:
            if _f1_subsample_n != n: _f1_subsample_idx = np.random.default_rng(1).choice(n, size=SUBSAMPLE_CAP, replace=False); _f1_subsample_n = n
            yt, yp = y_true[_f1_subsample_idx], y_pred[_f1_subsample_idx]
        else: yt, yp = y_true, y_pred
        precision, recall, thresholds = precision_recall_curve(yt, yp)
        f1_vals = (2.0 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
        raw_f1 = float(np.max(f1_vals)) if len(f1_vals) > 0 else 0.0
        # v6.5 FIX: Strictly return (name, value) tuple for LightGBM custom metrics
        return "f1", raw_f1
    except ValueError:
        return "f1", 0.0

def _resolve_model_type(requested: str) -> str:
    if requested in ("lightgbm",): return requested
    if requested == "auto":
        if LGBMClassifier is not None: return "lightgbm"
        raise ImportError("No GGBT installed.")
    raise ValueError(f"Unknown model type: {requested!r}")

def _fit_gbdt(x_train, y_train, model_type, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, seed, reg_alpha=0.1, x_eval=None, y_eval=None, gbdt_n_jobs=2):
    curve, best_iter = None, n_estimators
    if model_type == "lightgbm":
        if LGBMClassifier is None: raise ImportError("lightgbm not installed.")
        evals_result = {}; fit_params = {}; has_eval = x_eval is not None and y_eval is not None and len(y_eval) > 0
        model = LGBMClassifier(objective="binary", n_estimators=n_estimators, num_leaves=31, max_depth=-1, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, is_unbalance=True, min_child_samples=50, reg_alpha=reg_alpha, reg_lambda=1.0, random_state=seed, n_jobs=gbdt_n_jobs, verbose=-1)
        if has_eval:
            fit_params.update({"eval_set": [(x_eval, y_eval)], "eval_metric": ["f2", "auc", _f1_lgb], "callbacks": [lgb.early_stopping(150, verbose=False, first_metric_only=True), lgb.record_evaluation(evals_result)]})
        model.fit(x_train, y_train, **fit_params)
        if has_eval and evals_result: curve = {"iterations": list(range(1, len(evals_result["valid_0"]["f2"]) + 1)), "f2": evals_result["valid_0"]["f2"], "auc": evals_result["valid_0"]["auc"], "f1": evals_result["valid_0"]["f1"]}
        best_iter = getattr(model, "best_iteration_", n_estimators)
    return model, 0.0, curve, best_iter

def _make_cv_folds(valid_idx, n_folds, first_lag, min_calib=12, min_test=12, max_window_size=120):
    n_valid = len(valid_idx); required_overhead = first_lag + min_calib + min_test
    if n_valid <= required_overhead + 20: return []
    max_W_for_1_fold = n_valid - required_overhead
    if max_W_for_1_fold <= max_window_size: K = 1; W = max_W_for_1_fold
    else: K = math.ceil((n_valid - max_window_size - first_lag - min_calib) / min_test); W = n_valid - (K * min_test) - first_lag - min_calib
    while K > 1 and W < 20: K -= 1; W = n_valid - (K * min_test) - first_lag - min_calib
    if W < 20: return []
    folds = []
    for i in range(K):
        te_end = n_valid - (i * min_test); te_start = n_valid - ((i + 1) * min_test)
        ca_end = te_start; ca_start = ca_end - min_calib; tr_end = ca_start - first_lag; tr_start = tr_end - W
        folds.append((valid_idx[tr_start:tr_end], valid_idx[ca_start:ca_end], valid_idx[te_start:te_end]))
    folds.reverse(); print(f"    -> Strict Rolling: {len(folds)} folds, Fixed Window={W}w (100% data utilized)"); return folds

def train_one_fold(fold, train_idx, calib_idx, test_idx, adj_sources, adj_target, sector_labels, model_type, neg_ratio, history_lags, first_lag, calibration, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, save_dir, target_w, source_ws, seed, save_plots, shap_max_samples, rng, ablation_variants, eval_tail_frac=0.2, t_w=0, gbdt_n_jobs=2, reg_alpha=0.1, tickers=None):
    old_stdout = sys.stdout; sys.stdout = buffer = io.StringIO()
    try:
        train_idx = train_idx[train_idx >= first_lag]; calib_idx = calib_idx[calib_idx >= first_lag]; test_idx = test_idx[test_idx >= first_lag]
        if len(train_idx) < 20 or len(calib_idx) == 0 or len(test_idx) == 0: return [], [], buffer.getvalue()
        tail_n = max(10, int(len(train_idx) * eval_tail_frac)); train_early_idx, train_tail_idx = train_idx[:-tail_n], train_idx[-tail_n:]
        sector_to_id = {s: i for i, s in enumerate(sorted(set(sector_labels)))}; sector_ids = np.array([sector_to_id[s] for s in sector_labels], dtype=np.int32)
        feat_names_full = _feature_names(source_ws, target_w, history_lags, first_lag)
        print(f"\n  Fold {fold}  |  tr_early={len(train_early_idx)}  tr_tail={len(train_tail_idx)}  calib={len(calib_idx)}  test={len(test_idx)}  first_lag={first_lag}")
        print("  Building matrices ...", end=" ", flush=True)
        x_early, y_early = _build_train_matrix(train_early_idx, adj_sources, adj_target, sector_ids, history_lags, first_lag, target_w, neg_ratio, rng)
        x_tail, y_tail = _build_train_matrix(train_tail_idx, adj_sources, adj_target, sector_ids, history_lags, first_lag, target_w, neg_ratio, rng)
        x_train, y_train = np.concatenate([x_early, x_tail]), np.concatenate([y_early, y_tail])
        if len(y_train) == 0 or y_train.sum() == 0: print("empty."); return [], [], buffer.getvalue()
        print(f"train_early={len(y_early):,}  train_tail={len(y_tail):,}  total={len(y_train):,}  pos={int(y_train.sum()):,}")
        x_calib, y_calib = _build_eval_matrix(calib_idx, adj_sources, adj_target, sector_ids, history_lags, first_lag, target_w)
        x_test, y_test = _build_eval_matrix(test_idx, adj_sources, adj_target, sector_ids, history_lags, first_lag, target_w)
        print(f"calib={len(y_calib):,}  test={len(y_test):,}")
        masks = {abl: _get_ablation_mask(feat_names_full, abl, target_w) for abl in ablation_variants}
        fnames = {abl: [n for n, k in zip(feat_names_full, masks[abl]) if k] for abl in ablation_variants}
        smallest_source_ws = min(source_ws)
        yt_mp, yp_mp = _marginal_prior_baseline(adj_target, train_idx, test_idx)
        yt_ss, yp_ss = _short_scale_oracle_baseline(adj_sources[smallest_source_ws], adj_target, test_idx, first_lag)
        results, training_curves, model_cal_scores = [], [], [], []
        for abl in ablation_variants:
            mask, fn = masks[abl], fnames[abl]; label = "Full" if abl == "none" else "Pure Cross-Scale"
            xt_e, xt_t, xt_f, xc, xte = x_early[:, mask], x_tail[:, mask], x_train[:, mask], x_calib[:, mask], x_test[:, mask]
            model, spw, curve, best_iter = _fit_gbdt(xt_f, y_train, model_type, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, seed, reg_alpha=reg_alpha, x_eval=xt_t, y_eval=y_tail, gbdt_n_jobs=gbdt_n_jobs)
            print(f"    [{label}]  best_iter={best_iter}  feats={len(fn)}  reg_alpha={reg_alpha}")
            if curve is not None: training_curves.append((label, curve, best_iter))
            if abl == ablation_variants[0]:
                with open(Path(save_dir) / f"best_{model_type}_w{'_'.join(map(str, source_ws))}_to_w{target_w}_fold{fold}.pkl", "wb") as f: pickle.dump(model, f)
            ys_cal_raw, ys_test_raw = model.predict_proba(xc)[:, 1].astype(np.float32), model.predict_proba(xte)[:, 1].astype(np.float32)
            ys_test_cal, cal_method, cal_info = _calibrate_on_calib_eval_on_test(y_calib, ys_cal_raw, y_test, ys_test_raw, calibration)
            cal_obj, _ = _fit_calibrator(y_calib, ys_cal_raw, cal_method); ys_calib_cal = _apply_calibrator(cal_obj, cal_method, ys_cal_raw)
            best_thr, best_calib_f2 = _best_f2_threshold(y_calib, ys_calib_cal); calib_f2 = _compute_metrics_silent(y_calib, ys_calib_cal, threshold=best_thr)["f2"]
            print(f"    [{label}]  threshold={best_thr:.4f}  F2@thr(calib)={calib_f2:.4f}  method={cal_method}")
            gbdt_m = _compute_metrics(y_test, ys_test_cal, threshold=best_thr, name=f"{model_type.upper()} ({label})")
            baseline_prior, baseline_oracle = None, None
            if abl == ablation_variants[0]:
                _compute_metrics(yt_mp, yp_mp, threshold=best_thr, name="Marginal prior")
                _compute_metrics(yt_ss, yp_ss, threshold=best_thr, name=f"Oracle (A_w{smallest_source_ws})")
                baseline_prior = _compute_metrics_silent(yt_mp, yp_mp, threshold=best_thr); baseline_oracle = _compute_metrics_silent(yt_ss, yp_ss, threshold=best_thr)
            results.append({"fold": fold, "ablation": abl, "ablation_label": label, "source_ws": ",".join(map(str, source_ws)), "target_w": target_w, "n_features": len(fn), "best_iteration": best_iter, "calibration_method": cal_method, "brier_raw": cal_info["brier_raw"], "brier_calibrated": cal_info["brier_calibrated"], "n_train_steps": len(train_idx), "n_calib_steps": len(calib_idx), "n_test_steps": len(test_idx), "model_type": model_type, "gbdt": gbdt_m, "marginal_prior": baseline_prior, "short_scale_oracle": baseline_oracle})
            model_cal_scores[label] = ys_test_cal
            if save_plots:
                plots_dir = Path(save_dir) / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
                pfx = f"w{'_'.join(map(str, source_ws))}_to_w{target_w}_fold{fold}_{model_type}" + (f"_{abl}" if abl!="none" else "")
                _plot_feature_importance(model, fn, plots_dir, pfx); _plot_model_diagnostics(y_test, ys_test_cal, best_thr, plots_dir, pfx)
                if shap_max_samples > 0: n = min(shap_max_samples, len(xt_f)); shap_idx = rng.choice(len(xt_f), size=n, replace=False); _plot_shap_summary(model, xt_f[shap_idx], fn, plots_dir, pfx)
        if save_plots:
            plots_dir = Path(save_dir) / "plots"; plots_dir.mkdir(parents=True, exist_ok=True); pfx_base = f"w{'_'.join(map(str, source_ws))}_to_w{target_w}_fold{fold}"
            all_scores = {"Marginal Prior": yp_mp, f"Oracle (A_w{smallest_source_ws})": yp_ss}; all_scores.update(model_cal_scores); _plot_baseline_pr_comparison(y_test, all_scores, plots_dir, pfx_base)
        if save_plots and training_curves:
            plots_dir = Path(save_dir) / "plots"; plots_dir.mkdir(parents=True, exist_ok=True); _plot_training_curves(training_curves, plots_dir, f"w{'_'.join(map(str, source_ws))}_to_w{target_w}_fold{fold}_{model_type}")
        gc.collect(); return results, training_curves, buffer.getvalue()
    finally: sys.stdout = old_stdout

def run_training(source_ws, target_w, pkldir, model_type, neg_ratio, history_lags, calibration, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, save_dir, seed, save_plots, shap_max_samples, n_folds, min_calib_steps, min_test_steps, do_ablation, eval_tail_frac=0.2, parallel_folds=3, gbdt_n_jobs=2, max_window_size=120):
    source_ws = [ws for ws in source_ws if ws != target_w]
    if not source_ws: raise ValueError("source_ws must contain at least one window different from target_w")
    
    physical_cores = os.cpu_count() or 1
    if parallel_folds > 1 and gbdt_n_jobs == -1:
        print(f"\n  ⚠️  WARNING: --parallel_folds={parallel_folds} with --gbdt_n_jobs=-1 causes CPU thrashing. Auto-setting to 1.\n")
        gbdt_n_jobs = 1
    elif parallel_folds > 1 and gbdt_n_jobs > 1 and (parallel_folds * gbdt_n_jobs > physical_cores):
        print(f"\n  ⚠️  WARNING: {parallel_folds} folds x {gbdt_n_jobs} threads > {physical_cores} cores. Expect CPU thrashing.\n")
        
    if learning_rate < 0.05: print("\n  ⚠️  WARNING: LR < 0.05 is highly likely to cause premature early stopping.\n")
        
    chosen_mt = _resolve_model_type(model_type); first_lag = _min_safe_lag(target_w); reg_alpha = _get_scale_dependent_l1(target_w)
    src_label = ",".join(map(str, source_ws))
    print("\n" + "#" * 68)
    print(f"  CROSS-SCALE GBDT  A_[{src_label}] -> A_w{target_w}")
    print(f"  Model: {chosen_mt}  |  first_lag={first_lag}  |  reg_alpha={reg_alpha}  |  Max Window={max_window_size}w  |  Parallel Folds={parallel_folds}  |  GBDT Threads={gbdt_n_jobs}")
    print(f"  Objective: F2 Early Stopping (patience=150)  |  LR={learning_rate}  |  Max Trees={n_estimators}")
    print(f"  Features: Source+Neck (12/src) + Target (6) = {1 + (12 * len(source_ws) + 6) * history_lags} total")
    print("#" * 68)
    all_ws = list(set(source_ws + [target_w])); adj_dict, sector_labels, tickers_ordered = {}, None, None
    for w in all_ws: adj_dict[w], sector_labels, tickers_ordered = _load_adj_weekly(w, pkldir)
    t_w = list(adj_dict.values())[0].shape[0]; adj_sources = {ws: adj_dict[ws] for ws in source_ws}; adj_target = adj_dict[target_w]
    max_safe_week = t_w - math.ceil(target_w / WEEKLY_STRIDE); all_target_idx = np.arange(first_lag, max_safe_week)
    folds = _make_cv_folds(all_target_idx, n_folds, first_lag, min_calib_steps, min_test_steps, max_window_size)
    print(f"\n  Temporal CV: {len(folds)} folds (Rolling Window={max_window_size} weeks)")

    if save_plots and folds:
        plots_dir = Path(save_dir) / "plots"; plots_dir.mkdir(parents=True, exist_ok=True); run_label = f"w{'_'.join(map(str, source_ws))}_to_w{target_w}"
        _plot_dataset_timeline(all_target_idx, folds, t_w, first_lag, plots_dir, run_label)
        sector_to_id = {s: i for i, s in enumerate(sorted(set(sector_labels)))}; sector_ids = np.array([sector_to_id[s] for s in sector_labels], dtype=np.int32)
        feat_names_full = _feature_names(source_ws, target_w, history_lags, first_lag)
        x_snap, y_snap = _build_eval_matrix(folds[-1][2], adj_sources, adj_target, sector_ids, history_lags, first_lag, target_w); snapshot_rng = np.random.default_rng(seed); chosen_idx = _get_snapshot_indices(y_snap, snapshot_rng)
        _plot_feature_snapshot(x_snap, y_snap, feat_names_full, plots_dir, run_label, tickers_ordered, chosen_idx, target_w); _plot_feature_snapshot_standardized(x_snap, y_snap, feat_names_full, plots_dir, run_label, tickers_ordered, chosen_idx, target_w)

    ablation_variants = ["none"] + (["pure_cross_scale"] if do_ablation else [])
    all_results = []
    def _run_single_fold(fold_i_tuple):
        fold_i, (tr, ca, te) = fold_i_tuple; local_rng = np.random.default_rng(seed + fold_i)
        return train_one_fold(fold_i+1, tr, ca, te, adj_sources, adj_target, sector_labels, chosen_mt, neg_ratio, history_lags, first_lag, calibration, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, save_dir, target_w, source_ws, seed + fold_i, save_plots, shap_max_samples, local_rng, ablation_variants, eval_tail_frac, t_w, gbdt_n_jobs, reg_alpha, tickers_ordered)

    parallel_n = min(parallel_folds, len(folds))
    print(f"  Dispatching {parallel_n} workers...")
    fold_results_list = Parallel(n_jobs=parallel_n, backend="loky")(delayed(_run_single_fold)(item) for item in enumerate(folds))

    all_results = []
    for fold_res, _, log_str in fold_results_list:
        if log_str: print(log_str, end="", flush=True)
        all_results.extend(fold_res)
    return all_results

def _print_cv_summary(results):
    if not results: return
    src_label, tgt_w = results[0]["source_ws"], results[0]["target_w"]; abl_labels = sorted(set(r["ablation_label"] for r in results))
    print("\n" + "=" * 84); print(f"CV SUMMARY  A_[{src_label}] -> A_w{tgt_w}  (F2 Early Stopping, patience=150)"); print("=" * 84)
    for abl in abl_labels:
        subset = [r for r in results if r["ablation_label"] == abl]; print(f"\n  Ablation: {abl} ({subset[0].get('n_features','')} feats)")
        print(f"  {'AP':>7}  {'AUC':>7}  {'F1':>7}  {'F2':>7}  {'Prec':>7}  {'Rec':>7}  {'Brier':>10}  {'Iter':>5}")
        ap_l, auc_l, f1_l, f2_l, pr_l, rc_l, br_l, it_l = [], [], [], [], [], [], [], [], []
        for r in subset:
            m = r["gbdt"]; print(f"  F{r['fold']:>2d}  {m['ap']:>7.4f}  {m['auc']:>7.4f}  {m['f1']:>7.4f}  {m['f2']:>7.4f}  {m['prec']:>7.4f}  {m['rec']:>7.4f}  {m['brier']:>10.6f}  {r['best_iteration']:>5d}")
            ap_l.append(m["ap"]); auc_l.append(m["auc"]); f1_l.append(m["f1"]); f2_l.append(m["f2"]); pr_l.append(m["prec"]); rc_l.append(m["rec"]); br_l.append(m["brier"]); it_l.append(r["best_iteration"])
        def _s(v): a=np.array(v); val=a[~np.isnan(a)]; return (f"{val.mean():.4f}", f"{val.std():.4f}") if len(val)>0 else ("nan", "nan")
        am,as_ = _s(ap_l); um,us = _s(auc_l); fm,fs = _s(f1_l); f2m,f2s = _s(f2_l); pm,ps = _s(pr_l); rm,rs = _s(rc_l); bm,bs = _s(br_l); im,is_ = _s(it_l)
        print(f"  {'Mean':>5s}  {am:>7s}  {um:>7s}  {fm:>7s}  {f2m:>7s}  {pm:>7s}  {rm:>7s}  {bm:>10s}  {im:>5s}")
        print(f"  {'Std':>5s}  {as_:>7s}  {us:>7s}  {fs:>7s}  {f2s:>7s}  {ps:>7s}  {rs:>7s}  {bs:>10s}  {is_:>5s}")
        if abl == abl_labels[0]:
            prior, oracle = subset[0]["marginal_prior"], subset[0]["short_scale_oracle"]
            print(f"  {'Prior':>5s}  {prior['ap']:>7.4f}  {prior['auc']:>7.4f}  {prior['f1']:>7.4f}  {prior['f2']:>7.4f}  {prior['prec']:>7.4f}  {prior['rec']:>7.4f}  {prior['brier']:>10.6f}  {'':>5s}")
            print(f"  {'Oracle':>5s}  {oracle['ap']:>7.4f}  {oracle['auc']:>7.4f}  {oracle['f1']:>7.4f}  {oracle['f2']:>7.4f}  {oracle['prec']:>7.4f}  {oracle['rec']:>7.4f}  {oracle['brier']:>10.6f}  {'':>5s}")
    if len(abl_labels) > 1:
        print(f"\n  {'═' * 80}\n  ABLATION COMPARISON\n  {'═' * 80}"); full_f2 = None
        for abl in abl_labels:
            subset = [r for r in results if r["ablation_label"] == abl]; f2s = [r["gbdt"]["f2"] for r in subset if not np.isnan(r["gbdt"]["f2"])]
            if f2s: mean_f2 = np.mean(f2s); print(f"  {abl:<25s}  Mean F2 = {mean_f2:.4f}  (std = {np.std(f2s):.4f})")
            if abl == "Full": full_f2 = mean_f2
        if full_f2 is not None:
            pure_f2_list = [r["gbdt"]["f2"] for r in results if r["ablation_label"] == "Pure Cross-Scale" and not np.isnan(r["gbdt"]["f2"])]
            if pure_f2_list:
                pure_f2 = np.mean(pure_f2_list)
                if pure_f2 > full_f2: print("\n  >> WARNING: Pure Cross-Scale OUTPERFORMS Full model! Target features may be noise.")
                else: delta = ((full_f2 - pure_f2) / pure_f2) * 100 if pure_f2 > 0 else 0; print(f"\n  >> Full model outperforms Pure Cross-Scale by {delta:.1f}% relative F2.")

def _save_results_to_csv(results, out_path):
    if not results: return
    flat_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("gbdt", "marginal_prior", "short_scale_oracle")}
        for prefix, metrics in [("gbdt", r["gbdt"]), ("prior", r.get("marginal_prior") or {}), ("oracle", r.get("short_scale_oracle") or {})]:
            for mk, mv in metrics.items(): row[f"{prefix}_{mk}"] = mv
        flat_rows.append(row)
    if not flat_rows: return
    fieldnames = list(flat_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader(); writer.writerows(flat_rows)
    print(f"\n  Results saved to {out_path}")

# ─────────────────────────────────────────────────────────────────────────────────────
# Main Execution & Presets
# ─────────────────────────────────────────────────────────────────────────────

CROSS_SCALE_PRESETS = [
    {"source_ws": [35], "target_w": 70}, {"source_ws": [35], "target_w": 120}, {"source_ws": [35], "target_w": 180},
    {"source_ws": [70], "target_w": 35}, {"source_ws": [70], "target_w": 120}, {"source_ws": [120], "target_w": 35},
]

def main():
    parser = argparse.ArgumentParser(description="Cross-Scale GBDT Link Prediction (v6.5)")
    parser.add_argument("--pkldir", type=str, default=DEFAULT_PKL_DIR)
    parser.add_argument("--save_dir", type=str, default="./cross_scale_results_v6")
    parser.add_argument("--model_type", type=str, default="lightgbm", choices=["lightgbm", "auto"])
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--neg_ratio", type=int, default=3)
    parser.add_argument("--history_lags", type=int, default=1)
    parser.add_argument("--calibration", type=str, default="auto", choices=["none", "platt", "isotonic", "auto"])
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--min_calib_steps", type=int, default=12)
    parser.add_argument("--min_test_steps", type=int, default=12)
    parser.add_argument("--max_window_size", type=int, default=120)
    parser.add_argument("--eval_tail_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_plots", action="store_true", default=True)
    parser.add_argument("--no_plots", dest="save_plots", action="store_false")
    parser.add_argument("--shap_max_samples", type=int, default=5000)
    
    # v6.4 FIX: Option C (Hybrid) -> 3 folds x 2 threads = 6 cores total
    parser.add_argument("--parallel_folds", type=int, default=3, help="Number of CV folds to run in parallel via joblib.")
    parser.add_argument("--gbdt_n_jobs", type=int, default=2, help="Threads PER LightGBM instance. Keep at 1 when using --parallel_folds > 1.")
    parser.add_argument("--no_ablation", dest="do_ablation", action="store_false", default=True, help="Disable the Pure Cross-Scale ablation study")
    parser.add_argument("--source_ws", type=int, nargs="+", default=None)
    parser.add_argument("--target_w", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Run all possible single-source to single-target scale combinations automatically")
    
    args = parser.parse_args()

    save_base = Path(args.save_dir); save_base.mkdir(parents=True, exist_ok=True)

    if args.all:
        scales = list(EPSILON_CONFIGS.keys())
        presets = [{"source_ws": [src], "target_w": tgt} for src in scales for tgt in scales if src != tgt]
        print(f"\n  ⚙️  --all flag detected: Generated {len(presets)} cross-scale presets from {scales}")
    elif args.source_ws is not None and args.target_w is not None:
        presets = [{"source_ws": args.source_ws, "target_w": args.target_w}]
    else:
        presets = CROSS_SCALE_PRESETS

    all_final_results = []
    for preset in presets:
        src_ws = preset["source_ws"]; tgt_w = preset["target_w"]; src_label = "_".join(map(str, src_ws))
        preset_save_dir = save_base / f"w{src_label}_to_w{tgt_w}"; preset_save_dir.mkdir(parents=True, exist_ok=True)

        results = run_training(
            source_ws=src_ws, target_w=tgt_w, pkldir=args.pkldir, model_type=args.model_type,
            neg_ratio=args.neg_ratio, history_lags=args.history_lags, calibration=args.calibration,
            n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
            subsample=args.subsample, colsample_bytree=args.colsample_bytree, save_dir=str(preset_save_dir),
            seed=args.seed, save_plots=args.save_plots, shap_max_samples=args.shap_max_samples,
            n_folds=args.n_folds, min_calib_steps=args.min_calib_steps, min_test_steps=args.min_test_steps,
            do_ablation=args.do_ablation, eval_tail_frac=args.eval_tail_frac,
            parallel_folds=args.parallel_folds, gbdt_n_jobs=args.gbdt_n_jobs, max_window_size=args.max_window_size,
        )
        _print_cv_summary(results); _save_results_to_csv(results, preset_save_dir / "cv_results.csv"); all_final_results.extend(results); gc.collect()

    if len(presets) > 1:
        master_csv = save_base / "all_presets_cv_results.csv"; _save_results_to_csv(all_final_results, master_csv)
        print("\n" + "╔" + "═" * 82 + "╗"); print("║" + " MASTER SUMMARY ACROSS ALL PRESETS ".center(82) + "║"); print("╚" + "═" * 82 + "╝")
        grouped = {}
        for r in all_final_results:
            key = f"A_[{r['source_ws']}] -> w{r['target_w']} ({r['ablation_label']})"
            if key not in grouped: grouped[key] = []
            f2 = r["gbdt"]["f2"]
            if not np.isnan(f2): grouped[key].append(f2)
        print(f"\n  {'Preset':<40s}  {'Mean F2':>8s}  {'Std F2':>8s}  {'Count':>6s}"); print("  " + "-" * 64)
        for key, f2s in sorted(grouped.items()):
            if f2s: print(f"  {key:<40s}  {np.mean(f2s):>8.4f}  {np.std(f2s):>8.4f}  {len(f2s):>6d}")

    print("\n✅ All training complete.")

if __name__ == "__main__":
    main()