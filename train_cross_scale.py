"""
train_cross_scale.py
====================
Training and evaluation pipeline for cross-scale adjacency prediction.

Task
----
Predict A_wlong[t+1] (w=120 or w=180) from a window of L weekly
snapshots of A_w35[t-L+1 : t], using a Temporal GraphSAGE + GRU model.

Design decisions (with justifications)
---------------------------------------
Temporal resolution — WEEKLY (stride=5 trading days)
    Daily flip rates are 0.04–0.17% with imbalance 587:1 to 2390:1;
    unlearnable even with focal loss.  Weekly reduces imbalance to
    120:1–478:1 and keeps 315 snapshots — sufficient for time-series CV.

Loss — Focal loss (γ configurable per target)
    w=35  target (120:1): γ=2
    w=120 target (478:1): γ=3
    w=180 target (402:1): γ=3
    Standard BCE collapses to all-zeros prediction given this imbalance.

Pair sampling — positive-pair subgraph sampling at neg_ratio:1
    All positive pairs + fixed multiple of negatives per training step.
    Preserves full graph topology for neighbourhood aggregation while
    keeping loss computation balanced.

Cross-validation — TimeSeriesSplit (no shuffle, temporal ordering preserved)
    Identical to the ViT training pipeline in training_and_validation_functions.py.

Evaluation metrics (reported per fold)
    AUC-ROC, Average Precision (AP), F1 at threshold=0.5,
    Precision, Recall, Brier score.
    Also reports: improvement over persistence baseline and
    improvement over the multi-scale AND rule (predict 1 iff
    A_w35[t]=1 AND A_wlong[t]=1).

Usage
-----
    # Train on both target scales, all default settings:
    python train_cross_scale.py --all

    # Single target:
    python train_cross_scale.py --target 120
    python train_cross_scale.py --target 180

    # Override pkl directory:
    python train_cross_scale.py --all --pkldir /path/to/pkls

    # Quick smoke test (fewer epochs, small model):
    python train_cross_scale.py --target 120 --epochs 5 --embed-dim 16

Output
------
    best_gnn_w{target}_fold{k}.pt   — best checkpoint per fold
    gnn_cross_scale_results.csv     — per-fold metrics for all runs
    Console report with fold summaries and final cross-fold statistics.
"""

import argparse
import gc
import os
import pickle
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, brier_score_loss,
)

import parameters as p
from gnn_cross_scale import (
    CrossScaleGNN, FocalLoss, FocalLoss,
    build_node_features, adj_to_edge_index,
    sample_pairs, count_parameters,
)
from extract_distance_matrices import (
    SP500_TICKERS_457,
    reorder_by_gics,
    get_gics_sector_boundaries,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_W    = 35
SOURCE_EPS  = 0.1
TARGET_CONFIGS = {
    120: (0.4, p.GNN_FOCAL_GAMMA_W120),   # (epsilon, focal_gamma)
    180: (0.6, p.GNN_FOCAL_GAMMA_W180),
}

BAD_INDICES   = [6, 111, 128, 169, 170, 225]
N_STOCKS      = 457
WEEKLY_STRIDE = 5

DEFAULT_PKL_DIR = (
    "../Quasi_Differentiation_High_Temporal_Resolution_Cross_Correlations/Codes/"
    "Extract distance matrix (2017-2022) from pkl file"
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_adj_weekly(
    w: int,
    epsilon: float,
    pkldir: str,
    sector_labels: List[str],
) -> np.ndarray:
    """
    Load IQDw{w}.pkl → (T_w, N, N) binary float32 adjacency,
    weekly-strided and GICS-reordered.

    Returns
    -------
    adj_weekly : (T_w, N, N) float32  binary adjacency, weekly snapshots
    """
    path = os.path.join(pkldir, f"IQDw{w}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"pkl not found: {path}\nUse --pkldir to specify the correct directory."
        )
    print(f"  Loading IQDw{w}.pkl …", end=" ", flush=True)
    with open(path, "rb") as f:
        data = pickle.load(f).astype(np.float32)

    data = np.clip(data, -1.0, 1.0)
    dist = np.sqrt(2.0 * (1.0 - data)).astype(np.float32)
    del data

    adj = (dist <= epsilon).astype(np.float32)
    adj[:, np.arange(dist.shape[1]), np.arange(dist.shape[1])] = 0.0
    del dist

    adj = np.delete(adj, BAD_INDICES, axis=1)
    adj = np.delete(adj, BAD_INDICES, axis=2)
    assert adj.shape[1:] == (N_STOCKS, N_STOCKS), adj.shape

    # GICS reorder (axis 0 = time, axes 1,2 = stocks)
    _, tickers_gics, _ = reorder_by_gics(adj[0])
    # Build permutation from the reordering of the first snapshot
    ticker_to_pos = {t: i for i, t in enumerate(SP500_TICKERS_457)}
    perm = [ticker_to_pos[t] for t in tickers_gics]
    perm_arr = np.array(perm, dtype=np.intp)
    adj = adj[:, perm_arr, :][:, :, perm_arr]

    # Weekly stride
    adj_weekly = adj[::WEEKLY_STRIDE]
    print(f"done  daily={adj.shape}  weekly={adj_weekly.shape}")
    del adj
    return adj_weekly.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence & AND-rule baselines
# ─────────────────────────────────────────────────────────────────────────────

def _persistence_baseline(
    adj_long: np.ndarray,   # (T_w, N, N)
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict A_wlong[t+1] = A_wlong[t]."""
    triu = np.triu_indices(N_STOCKS, k=1)
    y_true = adj_long[test_idx][:, triu[0], triu[1]]            # (T_test, n_pairs)
    y_pred = adj_long[test_idx - 1][:, triu[0], triu[1]]        # (T_test, n_pairs)
    return y_true.ravel(), y_pred.ravel()


def _and_rule_baseline(
    adj_short: np.ndarray,  # (T_w, N, N)  A_w35
    adj_long:  np.ndarray,  # (T_w, N, N)  A_wlong
    test_idx:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict A_wlong[t+1] = 1  iff  A_w35[t]=1  AND  A_wlong[t]=1.
    (Multi-scale confidence-score rule from the design discussion.)
    """
    triu = np.triu_indices(N_STOCKS, k=1)
    y_true = adj_long[test_idx][:, triu[0], triu[1]]
    y_pred = (
        adj_short[test_idx - 1][:, triu[0], triu[1]] *
        adj_long[ test_idx - 1][:, triu[0], triu[1]]
    )
    return y_true.ravel(), y_pred.ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    name: str = "",
) -> Dict:
    y_pred = (y_score >= threshold).astype(np.int32)
    n_pos  = y_true.sum()

    if n_pos == 0 or n_pos == len(y_true):
        auc = ap = float("nan")
    else:
        auc = roc_auc_score(y_true, y_score)
        ap  = average_precision_score(y_true, y_score)

    f1    = f1_score(y_true, y_pred, zero_division=0)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_score)

    print(
        f"      {name:<28s}  AUC={auc:.4f}  AP={ap:.4f}  "
        f"F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Brier={brier:.6f}"
    )
    return dict(
        name=name, auc=auc, ap=ap, f1=f1,
        prec=prec, rec=rec, brier=brier,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single fold training + evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(
    fold:          int,
    train_idx:     np.ndarray,
    val_idx:       np.ndarray,
    adj_short:     np.ndarray,   # (T_w, N, N)  A_w35
    adj_long:      np.ndarray,   # (T_w, N, N)  A_wlong (target)
    sector_labels: List[str],
    model_cfg:     Dict,
    focal_gamma:   float,
    neg_ratio:     int,
    history_lags:  int,
    epochs:        int,
    lr:            float,
    weight_decay:  float,
    save_dir:      str,
    target_w:      int,
    rng:           np.random.Generator,
    device:        torch.device,
) -> Dict:
    """
    Train CrossScaleGNN on one time-series fold.

    train_idx / val_idx refer to the TARGET time step t+1.
    For each target time step t+1 in train_idx, the model sees
    A_w35[t-L+1 : t] as input.
    """
    print(f"\n{'─'*64}")
    print(f"  Fold {fold}  |  train T={len(train_idx)}  val T={len(val_idx)}")
    print(f"{'─'*64}")

    # Clamp indices: need at least L steps of history before first target
    L = history_lags
    train_idx = train_idx[train_idx >= L]
    val_idx   = val_idx[val_idx >= L]

    if len(train_idx) < 5:
        print("  ⚠  Too few training steps after lag clamp — skipping fold.")
        return {}

    # Model
    model = CrossScaleGNN(**model_cfg).to(device)
    print(f"  Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - 5, 1), eta_min=1e-6
    )
    criterion = FocalLoss(gamma=focal_gamma)

    best_val_ap  = -1.0
    best_path    = Path(save_dir) / f"best_gnn_w{target_w}_fold{fold}.pt"
    history      = {"train_loss": [], "val_ap": []}
    patience     = 15
    patience_ctr = 0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps    = 0

        # Shuffle training time steps each epoch
        t_perm = rng.permutation(train_idx)

        for t in t_perm:
            # Build input sequence: L snapshots of A_w35 ending at t-1
            # (we predict A_wlong[t], so we must not see A_wlong[t] in input)
            feat_seq  = []
            edge_seq  = []
            for lag in range(L - 1, -1, -1):   # oldest → newest
                step = int(t) - 1 - lag
                step = max(step, 0)
                snap_short = adj_short[step]       # (N, N)
                snap_long_prev = adj_long[max(step - 1, 0)]
                and_snapshot = np.logical_and(
                    snap_short > 0.5, snap_long_prev > 0.5
                ).astype(np.float32)
                x  = build_node_features(
                    sector_labels,
                    adj_snapshot=snap_short,
                    and_snapshot=and_snapshot,
                ).to(device)
                ei = adj_to_edge_index(snap_short).to(device)
                feat_seq.append(x)
                edge_seq.append(ei)

            # Sample pairs from A_wlong[t] (the target)
            pair_np, label_np = sample_pairs(
                adj_long[int(t)], neg_ratio=neg_ratio, rng=rng
            )
            if len(pair_np) == 0:
                continue

            pair_t  = torch.tensor(pair_np,  dtype=torch.long,  device=device)
            label_t = torch.tensor(label_np, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits = model(feat_seq, edge_seq, pair_t)
            loss   = criterion(logits, label_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps    += 1

            # Free GPU memory
            del feat_seq, edge_seq, pair_t, label_t, logits, loss

        if epoch > 5:
            scheduler.step()

        avg_loss = epoch_loss / max(n_steps, 1)
        history["train_loss"].append(avg_loss)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_scores = []
        val_labels = []

        triu_i, triu_j = np.triu_indices(N_STOCKS, k=1)
        pair_all = np.stack([triu_i, triu_j], axis=1)

        with torch.no_grad():
            for t in val_idx:
                feat_seq = []
                edge_seq = []
                for lag in range(L - 1, -1, -1):
                    step = int(t) - 1 - lag
                    step = max(step, 0)
                    snap_short = adj_short[step]
                    snap_long_prev = adj_long[max(step - 1, 0)]
                    and_snapshot = np.logical_and(
                        snap_short > 0.5, snap_long_prev > 0.5
                    ).astype(np.float32)
                    x  = build_node_features(
                        sector_labels,
                        adj_snapshot=snap_short,
                        and_snapshot=and_snapshot,
                    ).to(device)
                    ei = adj_to_edge_index(snap_short).to(device)
                    feat_seq.append(x)
                    edge_seq.append(ei)

                # Score all pairs in chunks to avoid OOM
                chunk = 8192
                all_logits = []
                for start in range(0, len(pair_all), chunk):
                    pt = torch.tensor(
                        pair_all[start: start + chunk],
                        dtype=torch.long, device=device,
                    )
                    lg = model(feat_seq, edge_seq, pt)
                    all_logits.append(lg.cpu())
                    del pt, lg

                logits_np = torch.cat(all_logits).numpy()
                probs_np  = 1.0 / (1.0 + np.exp(-logits_np))   # sigmoid
                labels_np = adj_long[int(t)][triu_i, triu_j]

                val_scores.append(probs_np)
                val_labels.append(labels_np)
                del feat_seq, edge_seq

        val_scores_all = np.concatenate(val_scores)
        val_labels_all = np.concatenate(val_labels)

        n_pos = val_labels_all.sum()
        if n_pos > 0 and n_pos < len(val_labels_all):
            val_ap = average_precision_score(val_labels_all, val_scores_all)
        else:
            val_ap = 0.0
        history["val_ap"].append(val_ap)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"loss={avg_loss:.5f}  val_AP={val_ap:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            patience_ctr = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_ap": val_ap,
                "history": history,
            }, best_path)
        else:
            patience_ctr += 1
            if epoch > 10 and patience_ctr >= patience:
                print(f"\n  Early stopping at epoch {epoch}  "
                      f"(best val_AP={best_val_ap:.4f})")
                break

    print(f"\n  Best val AP: {best_val_ap:.4f}  →  {best_path}")

    # ── Final evaluation on val set ───────────────────────────────────────
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Re-score full val set with best checkpoint
    val_scores2, val_labels2 = [], []
    with torch.no_grad():
        for t in val_idx:
            feat_seq, edge_seq = [], []
            for lag in range(L - 1, -1, -1):
                step = max(int(t) - 1 - lag, 0)
                snap_short = adj_short[step]
                snap_long_prev = adj_long[max(step - 1, 0)]
                and_snapshot = np.logical_and(
                    snap_short > 0.5, snap_long_prev > 0.5
                ).astype(np.float32)
                x  = build_node_features(
                    sector_labels,
                    adj_snapshot=snap_short,
                    and_snapshot=and_snapshot,
                ).to(device)
                ei = adj_to_edge_index(snap_short).to(device)
                feat_seq.append(x)
                edge_seq.append(ei)

            all_logits = []
            for start in range(0, len(pair_all), 8192):
                pt = torch.tensor(
                    pair_all[start: start + 8192],
                    dtype=torch.long, device=device,
                )
                all_logits.append(model(feat_seq, edge_seq, pt).cpu())
                del pt
            logits_np = torch.cat(all_logits).numpy()
            probs_np  = 1.0 / (1.0 + np.exp(-logits_np))
            val_scores2.append(probs_np)
            val_labels2.append(adj_long[int(t)][triu_i, triu_j])
            del feat_seq, edge_seq

    ys = np.concatenate(val_scores2)
    yt = np.concatenate(val_labels2)

    print(f"\n  Evaluation on validation set (fold {fold}):")
    gnn_metrics  = _compute_metrics(yt, ys,  name="GNN (CrossScaleGNN)")

    # Persistence baseline
    yt_p, yp_p  = _persistence_baseline(adj_long, val_idx[val_idx >= 1])
    yt_p = yt_p[:len(yt_p)]
    pers_metrics = _compute_metrics(yt_p, yp_p, name="Persistence A_wlong[t]")

    # AND-rule baseline
    yt_a, yp_a  = _and_rule_baseline(adj_short, adj_long, val_idx[val_idx >= 1])
    and_metrics  = _compute_metrics(yt_a, yp_a, name="AND-rule (A_w35 & A_wlong)")

    # Improvement over persistence
    if not np.isnan(gnn_metrics["ap"]) and not np.isnan(pers_metrics["ap"]):
        delta_ap = gnn_metrics["ap"] - pers_metrics["ap"]
        print(f"\n  Δ AP  vs persistence : {delta_ap:+.4f}")
        if delta_ap > 0:
            print("  ✓  GNN beats persistence on Average Precision")
        else:
            print("  ✗  GNN does not beat persistence — check loss/imbalance settings")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "fold": fold,
        "best_val_ap": best_val_ap,
        "gnn": gnn_metrics,
        "persistence": pers_metrics,
        "and_rule": and_metrics,
        "history": history,
        "best_path": str(best_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-fold cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(
    target_w:     int,
    pkldir:       str,
    model_cfg:    Dict,
    focal_gamma:  float,
    neg_ratio:    int,
    history_lags: int,
    epochs:       int,
    lr:           float,
    weight_decay: float,
    n_splits:     int,
    save_dir:     str,
    seed:         int,
) -> List[Dict]:
    """
    Full multi-fold time-series cross-validation for one target scale.
    """
    target_eps, _ = TARGET_CONFIGS[target_w]
    rng    = np.random.default_rng(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and p.USE_GPU else "cpu"
    )
    print(f"\n{'#'*68}")
    print(f"  CROSS-SCALE GNN  A_w{SOURCE_W} → A_w{target_w} (ε={target_eps})")
    print(f"  Device: {device}  |  Folds: {n_splits}  |  Epochs: {epochs}")
    print(f"{'#'*68}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading adjacency matrices …")

    # Need GICS sector labels — derive from any snapshot
    tmp_path = os.path.join(pkldir, f"IQDw{SOURCE_W}.pkl")
    with open(tmp_path, "rb") as f:
        tmp = pickle.load(f).astype(np.float32)
    tmp = np.clip(tmp, -1.0, 1.0)
    tmp_dist = np.sqrt(2.0 * (1.0 - tmp[0])).astype(np.float32)
    tmp_adj  = (tmp_dist <= SOURCE_EPS).astype(np.float32)
    del tmp, tmp_dist
    tmp_adj = np.delete(tmp_adj, BAD_INDICES, axis=0)
    tmp_adj = np.delete(tmp_adj, BAD_INDICES, axis=1)
    _, tickers_gics, sector_labels = reorder_by_gics(tmp_adj)
    del tmp_adj

    adj_short = _load_adj_weekly(SOURCE_W, SOURCE_EPS, pkldir, sector_labels)
    adj_long  = _load_adj_weekly(target_w, target_eps,  pkldir, sector_labels)

    T_w = adj_short.shape[0]
    assert adj_long.shape[0] == T_w, "Snapshot count mismatch between scales"

    print(f"\n  Weekly snapshots: {T_w}")
    print(f"  Stocks: {N_STOCKS}  |  History lags: {history_lags}")
    print(f"  Model: {count_parameters(CrossScaleGNN(**model_cfg)):,} parameters\n")

    # ── Single expanding-window split (deployment-style holdout) ──────────
    # Split on target indices (1 … T_w-1); index 0 has no predecessor.
    target_indices = np.arange(1, T_w)
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    holdout_steps = max(1, len(target_indices) // (n_splits + 1))
    split_at = len(target_indices) - holdout_steps
    if split_at <= 0:
        raise ValueError("Not enough weekly snapshots for holdout split.")

    train_idx = target_indices[:split_at]
    val_idx = target_indices[split_at:]
    print(
        f"  Expanding-window split: train={len(train_idx)} steps, "
        f"holdout={len(val_idx)} steps"
    )

    result = train_one_fold(
        fold=1,
        train_idx=train_idx,
        val_idx=val_idx,
        adj_short=adj_short,
        adj_long=adj_long,
        sector_labels=sector_labels,
        model_cfg=model_cfg,
        focal_gamma=focal_gamma,
        neg_ratio=neg_ratio,
        history_lags=history_lags,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        save_dir=save_dir,
        target_w=target_w,
        rng=rng,
        device=device,
    )
    return [result] if result else []


# ─────────────────────────────────────────────────────────────────────────────
# Summary and CSV output
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(target_w: int, fold_results: List[Dict]):
    if not fold_results:
        print("No fold results to summarise.")
        return

    print(f"\n{'═'*68}")
    print(f"CROSS-FOLD SUMMARY  A_w{SOURCE_W} → A_w{target_w}")
    print(f"{'═'*68}")

    header = f"  {'Fold':>4}  {'AP(GNN)':>9}  {'AP(persist)':>12}  "
    header += f"{'AP(AND)':>9}  {'ΔAP':>7}  {'AUC':>7}"
    print(header)
    print("  " + "─" * 62)

    ap_gnns, ap_pers, deltas = [], [], []
    for r in fold_results:
        gnn_ap  = r["gnn"].get("ap",  float("nan"))
        pers_ap = r["persistence"].get("ap", float("nan"))
        and_ap  = r["and_rule"].get("ap", float("nan"))
        gnn_auc = r["gnn"].get("auc", float("nan"))
        delta   = gnn_ap - pers_ap if not (
            np.isnan(gnn_ap) or np.isnan(pers_ap)
        ) else float("nan")
        ap_gnns.append(gnn_ap)
        ap_pers.append(pers_ap)
        deltas.append(delta)
        print(
            f"  {r['fold']:>4}  {gnn_ap:>9.4f}  {pers_ap:>12.4f}  "
            f"{and_ap:>9.4f}  {delta:>+7.4f}  {gnn_auc:>7.4f}"
        )

    mean_ap  = float(np.nanmean(ap_gnns))
    mean_del = float(np.nanmean(deltas))
    print("  " + "─" * 62)
    print(
        f"  {'mean':>4}  {mean_ap:>9.4f}  {float(np.nanmean(ap_pers)):>12.4f}  "
        f"{'':>9}  {mean_del:>+7.4f}"
    )
    print(f"\n  Mean GNN AP : {mean_ap:.4f}")
    if mean_del > 0:
        print(f"  ✓  GNN beats persistence by mean ΔAP={mean_del:+.4f}")
    else:
        print(f"  ✗  GNN does NOT beat persistence (mean ΔAP={mean_del:+.4f})")
        print("     → Consider increasing focal γ, neg_ratio, or epochs.")
    print(f"{'═'*68}\n")


def _save_csv(all_results: List[Dict], target_w: int, path: str):
    rows = []
    for r in all_results:
        for model_key in ["gnn", "persistence", "and_rule"]:
            m = r.get(model_key, {})
            rows.append({
                "target_w": target_w,
                "fold":     r.get("fold", ""),
                "model":    model_key,
                "ap":       m.get("ap",    ""),
                "auc":      m.get("auc",   ""),
                "f1":       m.get("f1",    ""),
                "prec":     m.get("prec",  ""),
                "rec":      m.get("rec",   ""),
                "brier":    m.get("brier", ""),
            })

    write_header = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"Results appended → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train temporal GNN for cross-scale adjacency prediction.\n"
            "Source: A_w35 (w=35, ε=0.1)  →  Target: A_w120 or A_w180\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all",    action="store_true",
                        help="Train for both target scales (w=120 and w=180).")
    parser.add_argument("--target", type=int, choices=[120, 180], default=None,
                        help="Single target scale: 120 or 180.")
    parser.add_argument("--pkldir", type=str, default=None,
                        help="Directory containing IQDw{w}.pkl files.")
    parser.add_argument("--epochs", type=int,
                        default=p.GNN_EPOCHS,
                        help=f"Training epochs per fold (default: {p.GNN_EPOCHS}).")
    parser.add_argument("--embed-dim", type=int,
                        default=p.GNN_EMBED_DIM,
                        help=f"Node embedding / GRU dimension (default: {p.GNN_EMBED_DIM}).")
    parser.add_argument("--lags", type=int,
                        default=p.GNN_HISTORY_LAGS,
                        help=f"History lag steps (default: {p.GNN_HISTORY_LAGS}).")
    parser.add_argument("--neg-ratio", type=int,
                        default=p.GNN_NEG_RATIO,
                        help=f"Negative:positive ratio (default: {p.GNN_NEG_RATIO}).")
    parser.add_argument("--lr", type=float,
                        default=p.GNN_LR,
                        help=f"Learning rate (default: {p.GNN_LR}).")
    parser.add_argument("--n-splits", type=int,
                        default=p.GNN_N_SPLITS,
                        help=f"Number of CV folds (default: {p.GNN_N_SPLITS}).")
    parser.add_argument("--save-dir", type=str, default=".",
                        help="Directory for model checkpoints.")
    parser.add_argument("--seed", type=int, default=p.RANDOM_SEED)
    args = parser.parse_args()

    if not args.all and args.target is None:
        parser.error("Specify --all or --target {120|180}.")

    pkldir  = args.pkldir or DEFAULT_PKL_DIR
    targets = [120, 180] if args.all else [args.target]

    # Pre-check pkl files
    needed = set([SOURCE_W] + targets)
    missing = [
        os.path.join(pkldir, f"IQDw{w}.pkl")
        for w in needed
        if not os.path.isfile(os.path.join(pkldir, f"IQDw{w}.pkl"))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing pkl files:\n" + "\n".join(f"  {m}" for m in missing)
        )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "gnn_cross_scale_results.csv")

    for tw in targets:
        _, focal_gamma = TARGET_CONFIGS[tw]

        model_cfg = dict(
            in_dim         = 13,
            sage_hidden    = args.embed_dim,
            embed_dim      = args.embed_dim,
            gru_dim        = args.embed_dim,
            decoder_hidden = args.embed_dim // 2,
            dropout        = p.GNN_DROPOUT,
        )

        fold_results = run_cross_validation(
            target_w     = tw,
            pkldir       = pkldir,
            model_cfg    = model_cfg,
            focal_gamma  = focal_gamma,
            neg_ratio    = args.neg_ratio,
            history_lags = args.lags,
            epochs       = args.epochs,
            lr           = args.lr,
            weight_decay = p.GNN_WEIGHT_DECAY,
            n_splits     = args.n_splits,
            save_dir     = args.save_dir,
            seed         = args.seed,
        )

        _print_summary(tw, fold_results)
        if fold_results:
            _save_csv(fold_results, tw, csv_path)

    print("\nAll runs complete.")
