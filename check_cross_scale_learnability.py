"""
check_cross_scale_learnability.py
==================================
Diagnostic script to assess whether a machine / deep learning model can
meaningfully predict the w=120 or w=180 adjacency matrices from the w=35
distance matrix.

Four complementary lenses are applied:

  1. MUTUAL INFORMATION (MI)
       Point-wise MI between each (i,j) edge in A_w35 and the same edge in
       A_w120 / A_w180.  MI > 0 means the short-scale edge state carries
       information about the long-scale edge state.

  2. CROSS-SCALE EDGE OVERLAP & CONDITIONAL STATISTICS
       P(A_wlong=1 | A_wshort=1) vs P(A_wlong=1 | A_wshort=0).
       Lift = P(A_wlong=1 | A_wshort=1) / P(A_wlong=1).
       A lift >> 1 means the short-scale graph is a useful predictor.

  3. TEMPORAL LEAD-LAG CORRELATION
       For every pair (i,j) compute the Pearson correlation between
       A_w35[t] and A_wlong[t+lag] for lags 0,1,…,max_lag (weekly snapshots).
       A peak at lag > 0 means A_w35 *leads* A_wlong — causally useful.

  4. BASELINE vs NAIVE MODEL COMPARISON (on a held-out test split)
       Three predictors are compared on the task
           "predict A_wlong[t+1] from A_w35[t]":

         a) All-zeros baseline  — predict no edges ever
         b) Persistence baseline — predict A_wlong[t+1] = A_wlong[t]
         c) Short-scale oracle  — predict A_wlong[t+1] = threshold(A_w35[t])
            (optimal threshold chosen on the train split)
         d) Logistic regression — trained pair-wise on A_w35[t] → A_wlong[t+1]
            (tests whether a linear ML model can extract signal)
         e) Random Forest       — non-linear ML baseline

       Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, Brier Score.
       The key question: does any predictor beat the persistence baseline?

  5. FEATURE IMPORTANCE SANITY CHECK
       For the Logistic Regression / RF models, verify that the most
       predictive features are same-pair edges (diagonal of the feature
       importance matrix) — i.e. A_w35(i,j) predicts A_wlong(i,j) — rather
       than arbitrary cross-pair leakage.

Usage
-----
    # Both target scales, default settings:
    python check_cross_scale_learnability.py --all

    # Single target:
    python check_cross_scale_learnability.py --target 120
    python check_cross_scale_learnability.py --target 180

    # Adjust lag window and subsample size for speed:
    python check_cross_scale_learnability.py --all --max-lag 10 --n-pairs 2000

    # Override pkl directory:
    python check_cross_scale_learnability.py --all --pkldir /path/to/pkls

Output
------
    Cross-scale learnability report printed to stdout.
    Summary CSV saved to cross_scale_learnability_summary.csv.
    Lead-lag correlation plot saved to cross_scale_leadlag.png.
"""

import argparse
import os
import pickle
import warnings
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — safe on HPC nodes
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_W   = 35
SOURCE_EPS = 0.1
TARGET_CONFIGS = {
    120: 0.4,
    180: 0.6,
}

DEFAULT_PKL_DIR = (
    "../Quasi_Differentiation_High_Temporal_Resolution_Cross_Correlations/Codes/"
    "Extract distance matrix (2017-2022) from pkl file"
)

BAD_INDICES = [6, 111, 128, 169, 170, 225]
N_STOCKS    = 457
WEEKLY_STRIDE = 5          # sample every 5th trading day


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_adj(w: int, epsilon: float, pkldir: str) -> np.ndarray:
    """
    Load IQDw{w}.pkl → (T, 457, 457) binary float32 adjacency matrix.
    Correlation → distance → threshold at epsilon → remove 6 bad tickers.
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
    adj          = (dist <= epsilon).astype(np.float32)
    adj[:, np.arange(dist.shape[1]), np.arange(dist.shape[1])] = 0.0
    adj = np.delete(adj, BAD_INDICES, axis=1)
    adj = np.delete(adj, BAD_INDICES, axis=2)
    assert adj.shape[1:] == (N_STOCKS, N_STOCKS), adj.shape
    print(f"done  {adj.shape}")
    return adj


def _upper_tri_series(adj: np.ndarray) -> np.ndarray:
    """(T, N, N) → (T, n_pairs) upper-triangle time series."""
    triu = np.triu_indices(adj.shape[1], k=1)
    return adj[:, triu[0], triu[1]]


# ─────────────────────────────────────────────────────────────────────────────
# Lens 1 — Mutual Information
# ─────────────────────────────────────────────────────────────────────────────

def _pointwise_mi(x: np.ndarray, y: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute per-column point-wise MI between binary column vectors x[:,j]
    and y[:,j] across T time steps.

    MI = Σ_{a,b} P(X=a,Y=b) * log[ P(X=a,Y=b) / (P(X=a)*P(Y=b)) ]

    Returns array of shape (n_pairs,).
    """
    T, P = x.shape
    mi = np.zeros(P, dtype=np.float64)
    for a in [0, 1]:
        for b in [0, 1]:
            p_xy = ((x == a) & (y == b)).mean(axis=0) + eps
            p_x  = (x == a).mean(axis=0) + eps
            p_y  = (y == b).mean(axis=0) + eps
            mi  += p_xy * np.log(p_xy / (p_x * p_y))
    return mi.astype(np.float32)


def lens_mutual_information(
    src_series: np.ndarray,
    tgt_series: np.ndarray,
    label: str,
) -> dict:
    """
    Compute pair-wise MI between A_w35[t] and A_wlong[t] (same time step).
    Also compute MI at lag=1 (A_w35[t] vs A_wlong[t+1]).
    """
    print(f"\n  [MI] Computing mutual information ({label}) …")
    mi_lag0 = _pointwise_mi(src_series, tgt_series)
    mi_lag1 = _pointwise_mi(src_series[:-1], tgt_series[1:])

    result = {
        "mi_lag0_mean": float(mi_lag0.mean()),
        "mi_lag0_median": float(np.median(mi_lag0)),
        "mi_lag0_frac_positive": float((mi_lag0 > 1e-5).mean()),
        "mi_lag1_mean": float(mi_lag1.mean()),
        "mi_lag1_median": float(np.median(mi_lag1)),
        "mi_lag1_frac_positive": float((mi_lag1 > 1e-5).mean()),
    }

    print(f"    Same-time MI   : mean={result['mi_lag0_mean']:.5f}  "
          f"median={result['mi_lag0_median']:.5f}  "
          f"frac>0={result['mi_lag0_frac_positive']*100:.1f}%")
    print(f"    1-step lead MI : mean={result['mi_lag1_mean']:.5f}  "
          f"median={result['mi_lag1_median']:.5f}  "
          f"frac>0={result['mi_lag1_frac_positive']*100:.1f}%")
    print(f"    (MI=0 → no information; MI>0 → short-scale predicts long-scale)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Lens 2 — Edge overlap / conditional statistics
# ─────────────────────────────────────────────────────────────────────────────

def lens_conditional_stats(
    src_series: np.ndarray,
    tgt_series: np.ndarray,
    label: str,
) -> dict:
    """
    Compute:
      P(long=1)           — marginal edge probability in long-scale graph
      P(long=1 | short=1) — conditional probability given short edge active
      P(long=1 | short=0) — conditional probability given short edge inactive
      Lift = P(long=1|short=1) / P(long=1)

    All computed pair-wise (per column) and averaged.
    Also computed for the 1-step-ahead target.
    """
    print(f"\n  [COND] Conditional edge statistics ({label}) …")

    def _cond(src, tgt, lag_label):
        p_long     = tgt.mean()
        mask1      = src == 1
        mask0      = src == 0
        p_long_g1  = tgt[mask1].mean() if mask1.any() else float("nan")
        p_long_g0  = tgt[mask0].mean() if mask0.any() else float("nan")
        lift       = p_long_g1 / p_long if p_long > 0 else float("nan")
        print(f"      {lag_label:15s}  P(long=1)={p_long:.4f}  "
              f"P(long=1|short=1)={p_long_g1:.4f}  "
              f"P(long=1|short=0)={p_long_g0:.4f}  "
              f"Lift={lift:.2f}x")
        return p_long, p_long_g1, p_long_g0, lift

    p0, p1, pn1, l0 = _cond(src_series.flatten(), tgt_series.flatten(), "same-time")
    p0f, p1f, pn1f, l1 = _cond(
        src_series[:-1].flatten(), tgt_series[1:].flatten(), "1-step lead"
    )
    print(f"      Lift >> 1  → A_w35 is a strong predictor of A_wlong")
    print(f"      Lift ≈ 1   → A_w35 carries no useful information")

    return {
        "p_long": p0, "p_long_g_short1": p1, "p_long_g_short0": pn1,
        "lift_lag0": l0,
        "p_long_g_short1_lag1": p1f, "lift_lag1": l1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Lens 3 — Temporal lead-lag correlation
# ─────────────────────────────────────────────────────────────────────────────

def lens_leadlag(
    src_series: np.ndarray,
    tgt_series: np.ndarray,
    label: str,
    max_lag: int = 10,
    n_pairs_sample: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    For a random subsample of pairs, compute the mean Pearson correlation
    between A_w35[t] and A_wlong[t+lag] for lag = 0, 1, …, max_lag.

    A peak correlation at lag > 0 implies A_w35 leads A_wlong — potentially
    causally useful for prediction.
    """
    print(f"\n  [LAG] Lead-lag correlation up to lag={max_lag} ({label}) …")

    if rng is None:
        rng = np.random.default_rng(42)

    P = src_series.shape[1]
    n_sample = min(n_pairs_sample, P)
    idx = rng.choice(P, size=n_sample, replace=False)

    src_sub = src_series[:, idx].astype(np.float32)
    tgt_sub = tgt_series[:, idx].astype(np.float32)

    lags  = np.arange(0, max_lag + 1)
    means = np.zeros(len(lags))
    stds  = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        s = src_sub[:len(src_sub) - lag]       # (T-lag, sample)
        t = tgt_sub[lag:]                       # (T-lag, sample)

        # Pearson r per pair — avoid /0 for constant columns
        s_c = s - s.mean(axis=0, keepdims=True)
        t_c = t - t.mean(axis=0, keepdims=True)
        num = (s_c * t_c).sum(axis=0)
        den = np.sqrt((s_c**2).sum(axis=0) * (t_c**2).sum(axis=0)) + 1e-10
        r   = num / den

        means[i] = float(r.mean())
        stds[i]  = float(r.std())
        print(f"    lag={lag:2d}  mean_r={means[i]:.5f}  std={stds[i]:.5f}")

    peak_lag  = int(lags[means.argmax()])
    peak_corr = float(means.max())
    print(f"    → Peak correlation at lag={peak_lag}: r={peak_corr:.5f}")
    if peak_lag > 0:
        print(f"    → A_w35 LEADS A_{label.split('=')[1]} by {peak_lag} weekly snapshot(s) "
              f"(≈ {peak_lag * WEEKLY_STRIDE} trading days)")
    else:
        print(f"    → No detectable lead: peak at lag=0 (contemporaneous only)")

    return {"lags": lags, "mean_r": means, "std_r": stds,
            "peak_lag": peak_lag, "peak_corr": peak_corr}


# ─────────────────────────────────────────────────────────────────────────────
# Lens 4 — Baseline vs ML model comparison
# ─────────────────────────────────────────────────────────────────────────────

def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             y_score: Optional[np.ndarray] = None,
                             name: str = "") -> dict:
    """Compute Accuracy, Precision, Recall, F1, Brier, AUC-ROC."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, brier_score_loss, roc_auc_score,
    )

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_score if y_score is not None else y_pred)
    try:
        auc = roc_auc_score(y_true, y_score if y_score is not None else y_pred)
    except Exception:
        auc = float("nan")

    print(f"      {name:<25s}  "
          f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
          f"F1={f1:.4f}  AUC={auc:.4f}  Brier={brier:.6f}")
    return dict(name=name, acc=acc, prec=prec, rec=rec, f1=f1, auc=auc, brier=brier)


def lens_ml_baseline(
    src_weekly: np.ndarray,   # (T_w, n_pairs)  A_w35 weekly snapshots
    tgt_weekly: np.ndarray,   # (T_w, n_pairs)  A_wlong weekly snapshots
    label: str,
    n_pairs_sample: int = 3000,
    train_frac: float = 0.7,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Predict A_wlong[t+1] from A_w35[t].

    Subsamples n_pairs_sample pairs to keep memory/compute tractable.
    Each row is one (time-step, pair) observation:
        feature  = A_w35[t, pair]   (binary {0,1})
        label    = A_wlong[t+1, pair] (binary {0,1})

    Models:
        a) All-zeros baseline
        b) Persistence baseline: predict A_wlong[t+1] = A_wlong[t]
        c) Threshold oracle: predict A_wlong[t+1] = 1 if A_w35[t] >= 0.5
           (equivalent to: predict 1 whenever A_w35 edge is active)
        d) Logistic Regression (sklearn)
        e) Random Forest (sklearn, 50 trees, fast)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    print(f"\n  [ML] Baseline vs ML model comparison ({label}) …")

    if rng is None:
        rng = np.random.default_rng(42)

    T_w = src_weekly.shape[0]
    P   = src_weekly.shape[1]
    n_s = min(n_pairs_sample, P)
    idx = rng.choice(P, size=n_s, replace=False)

    # Build (T_w - 1) x n_s arrays
    X_raw = src_weekly[:-1][:, idx].astype(np.float32)   # A_w35[t]
    Y_raw = tgt_weekly[1:][:, idx].astype(np.float32)    # A_wlong[t+1]
    Yp    = tgt_weekly[:-1][:, idx].astype(np.float32)   # A_wlong[t]  (persistence)

    T_pred = X_raw.shape[0]
    split  = int(T_pred * train_frac)
    if split < 10 or T_pred - split < 5:
        print("    ⚠  Too few snapshots for train/test split — skipping ML lens")
        return {}

    # Flatten: each (time, pair) → one sample
    X_train_flat = X_raw[:split].reshape(-1, 1)
    Y_train_flat = Y_raw[:split].reshape(-1)
    X_test_flat  = X_raw[split:].reshape(-1, 1)
    Y_test_flat  = Y_raw[split:].reshape(-1)
    Yp_test_flat = Yp[split:].reshape(-1)

    pos_rate = Y_train_flat.mean()
    print(f"    Train samples : {len(Y_train_flat):,}  "
          f"(T={split}, pairs={n_s},  pos_rate={pos_rate:.4f})")
    print(f"    Test  samples : {len(Y_test_flat):,}  "
          f"(T={T_pred-split})")

    results = []

    # a) All-zeros
    r = _classification_metrics(
        Y_test_flat, np.zeros_like(Y_test_flat),
        y_score=np.zeros_like(Y_test_flat),
        name="All-zeros"
    )
    results.append(r)

    # b) Persistence (A_wlong[t] → A_wlong[t+1])
    r = _classification_metrics(
        Y_test_flat, Yp_test_flat,
        y_score=Yp_test_flat,
        name="Persistence (A_wlong[t])"
    )
    results.append(r)

    # c) Short-scale oracle (use A_w35[t] directly)
    r = _classification_metrics(
        Y_test_flat, X_test_flat.ravel(),
        y_score=X_test_flat.ravel(),
        name="Short-scale (A_w35[t])"
    )
    results.append(r)

    # d) Logistic Regression
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_flat)
    X_te_s = scaler.transform(X_test_flat)
    cw = "balanced" if 0.01 < pos_rate < 0.99 else None
    try:
        lr_model = LogisticRegression(
            C=1.0, class_weight=cw, max_iter=500, solver="lbfgs"
        )
        lr_model.fit(X_tr_s, Y_train_flat)
        lr_pred  = lr_model.predict(X_te_s)
        lr_prob  = lr_model.predict_proba(X_te_s)[:, 1]
        r = _classification_metrics(Y_test_flat, lr_pred, lr_prob,
                                    name="Logistic Regression")
        results.append(r)
    except Exception as e:
        print(f"      LogReg failed: {e}")

    # e) Random Forest
    try:
        rf_model = RandomForestClassifier(
            n_estimators=50, max_depth=4, class_weight=cw,
            n_jobs=-1, random_state=42
        )
        rf_model.fit(X_train_flat, Y_train_flat)
        rf_pred = rf_model.predict(X_test_flat)
        rf_prob = rf_model.predict_proba(X_test_flat)[:, 1]
        r = _classification_metrics(Y_test_flat, rf_pred, rf_prob,
                                    name="Random Forest (d=4)")
        results.append(r)
    except Exception as e:
        print(f"      RandomForest failed: {e}")

    # Verdict
    best_f1_model = max(results[2:], key=lambda x: x.get("f1", 0), default=None)
    persist_f1    = results[1]["f1"]
    print(f"\n    Persistence F1    : {persist_f1:.4f}")
    if best_f1_model:
        ml_f1 = best_f1_model.get("f1", 0)
        delta  = ml_f1 - persist_f1
        print(f"    Best ML F1        : {ml_f1:.4f}  ({best_f1_model['name']})")
        if delta > 0.01:
            print(f"    ✓  ML BEATS persistence by ΔF1={delta:.4f} — "
                  f"cross-scale prediction is feasible")
        elif delta > 0:
            print(f"    △  ML marginally better (ΔF1={delta:.4f}) — "
                  f"weak but detectable signal")
        else:
            print(f"    ✗  ML does NOT beat persistence (ΔF1={delta:.4f}) — "
                  f"negligible cross-scale signal at this granularity")

    return {"model_results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Lens 5 — Multi-feature ML (use richer feature set)
# ─────────────────────────────────────────────────────────────────────────────

def lens_ml_multifeature(
    src_weekly: np.ndarray,   # (T_w, n_pairs)  A_w35 weekly snapshots
    tgt_weekly: np.ndarray,   # (T_w, n_pairs)  A_wlong weekly snapshots
    label: str,
    n_pairs_sample: int = 2000,
    history_lags: int = 4,
    train_frac: float = 0.7,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Richer feature set: for each pair (i,j) and each time step t, the
    feature vector is:

        [ A_w35[t],  A_w35[t-1],  …,  A_w35[t-L+1],
          A_wlong[t],  A_wlong[t-1],  …,  A_wlong[t-L+1] ]

    This tests whether temporal context from both scales improves
    prediction of A_wlong[t+1] beyond a single-step feature.

    A Random Forest is trained on the flattened (time × pair) dataset.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss

    print(f"\n  [ML-MULTI] Multi-feature cross-scale prediction ({label}) …")

    if rng is None:
        rng = np.random.default_rng(42)

    T_w = src_weekly.shape[0]
    P   = src_weekly.shape[1]
    n_s = min(n_pairs_sample, P)
    idx = rng.choice(P, size=n_s, replace=False)

    src = src_weekly[:, idx].astype(np.float32)   # (T_w, n_s)
    tgt = tgt_weekly[:, idx].astype(np.float32)   # (T_w, n_s)

    L = history_lags
    # Build samples from t = L to T_w - 1
    T_start = L
    T_end   = T_w - 1   # predict t+1 up to T_w

    if T_end - T_start < 20:
        print("    ⚠  Too few snapshots for multi-feature test — skipping")
        return {}

    rows = []
    labels_list = []
    for t in range(T_start, T_end):
        feats_src = src[t - L + 1: t + 1].T   # (n_s, L)
        feats_tgt = tgt[t - L + 1: t + 1].T   # (n_s, L)
        feats     = np.hstack([feats_src, feats_tgt])  # (n_s, 2L)
        rows.append(feats)
        labels_list.append(tgt[t + 1])          # A_wlong[t+1]

    X_all = np.vstack(rows)                              # (n_samples, 2L)
    Y_all = np.concatenate(labels_list).astype(np.int32) # (n_samples,)

    split = int(len(Y_all) * train_frac)
    X_tr, X_te = X_all[:split], X_all[split:]
    Y_tr, Y_te = Y_all[:split], Y_all[split:]

    pos_rate = Y_tr.mean()
    print(f"    Feature dim   : {X_all.shape[1]}  (2×{L} lags, src+tgt)")
    print(f"    Train samples : {len(Y_tr):,}  (pos_rate={pos_rate:.4f})")
    print(f"    Test  samples : {len(Y_te):,}")

    if pos_rate < 1e-4 or pos_rate > 1 - 1e-4:
        print("    ⚠  Degenerate class distribution — skipping")
        return {}

    try:
        cw = {0: 1.0, 1: (1.0 - pos_rate) / max(pos_rate, 1e-6)}
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        gb.fit(X_tr, Y_tr)
        prob  = gb.predict_proba(X_te)[:, 1]
        pred  = (prob >= 0.5).astype(np.int32)

        f1  = f1_score(Y_te, pred, zero_division=0)
        auc = roc_auc_score(Y_te, prob) if len(np.unique(Y_te)) > 1 else float("nan")
        brier = brier_score_loss(Y_te, prob)

        # Feature importance: first L features are A_w35 lags,
        # next L features are A_wlong lags
        imp = gb.feature_importances_
        imp_src = imp[:L].sum()
        imp_tgt = imp[L:].sum()
        print(f"    GBM  F1={f1:.4f}  AUC={auc:.4f}  Brier={brier:.6f}")
        print(f"    Feature importance: A_w35 lags={imp_src:.3f}  "
              f"A_wlong lags={imp_tgt:.3f}")
        print(f"    (A_w35 importance > 0 → short-scale carries predictive signal "
              f"beyond A_wlong history alone)")

        return {
            "f1": f1, "auc": auc, "brier": brier,
            "imp_src": float(imp_src), "imp_tgt": float(imp_tgt),
        }
    except Exception as e:
        print(f"    GBM failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Plot: lead-lag correlation
# ─────────────────────────────────────────────────────────────────────────────

def plot_leadlag(
    results_120: Optional[dict],
    results_180: Optional[dict],
    save_path: str = "cross_scale_leadlag.png",
):
    fig, ax = plt.subplots(figsize=(8, 4))

    if results_120 is not None:
        lags = results_120["lags"]
        ax.errorbar(
            lags, results_120["mean_r"],
            yerr=results_120["std_r"],
            marker="o", label="A_w35 → A_w120  (ε=0.4)",
            color="#2196F3", capsize=3,
        )
    if results_180 is not None:
        lags = results_180["lags"]
        ax.errorbar(
            lags, results_180["mean_r"],
            yerr=results_180["std_r"],
            marker="s", linestyle="--",
            label="A_w35 → A_w180  (ε=0.6)",
            color="#FF5722", capsize=3,
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Lag  (weekly snapshots,  1 = 5 trading days)")
    ax.set_ylabel("Mean Pearson r  (A_w35[t] vs A_wlong[t+lag])")
    ax.set_title(
        "Cross-Scale Lead-Lag Correlation\n"
        "A_w35 (w=35, ε=0.1) → A_wlong\n"
        "Peak at lag > 0  ⟹  A_w35 leads A_wlong  (causal predictive signal)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nLead-lag plot saved → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table + CSV
# ─────────────────────────────────────────────────────────────────────────────

def _print_verdict(label: str, mi: dict, cond: dict, lag: dict, ml: dict):
    print(f"\n{'═'*70}")
    print(f"OVERALL VERDICT: A_w35 → {label}")
    print(f"{'═'*70}")

    scores = []

    # MI signal
    if mi.get("mi_lag1_mean", 0) > 1e-4:
        scores.append("MI: detectable information (lag-1 MI > 0)")
    else:
        scores.append("MI: essentially zero information")

    # Lift
    lift = cond.get("lift_lag1", 1.0)
    if lift > 2.0:
        scores.append(f"Lift={lift:.2f}x: strong conditional enrichment")
    elif lift > 1.2:
        scores.append(f"Lift={lift:.2f}x: moderate enrichment")
    else:
        scores.append(f"Lift={lift:.2f}x: weak enrichment (≈ marginal)")

    # Lead-lag
    if lag.get("peak_lag", 0) > 0:
        scores.append(
            f"Lead-lag: A_w35 leads by {lag['peak_lag']} snapshot(s), "
            f"r={lag['peak_corr']:.4f}"
        )
    else:
        scores.append("Lead-lag: no detectable lead (peak at lag=0)")

    # ML
    ml_results = ml.get("model_results", [])
    if ml_results:
        persist_f1 = next(
            (r["f1"] for r in ml_results if "Persistence" in r["name"]), 0.0
        )
        best_ml    = max(
            (r for r in ml_results if "Persistence" not in r["name"]
             and "All-zeros" not in r["name"]),
            key=lambda r: r.get("f1", 0), default=None,
        )
        if best_ml:
            delta = best_ml["f1"] - persist_f1
            if delta > 0.01:
                scores.append(
                    f"ML: beats persistence ΔF1={delta:.4f} "
                    f"({best_ml['name']})"
                )
            else:
                scores.append(f"ML: does not beat persistence (ΔF1={delta:.4f})")

    for s in scores:
        print(f"  • {s}")

    positive = sum(
        1 for s in scores
        if any(k in s for k in ["detectable", "strong", "moderate", "beats", "leads"])
    )
    total = len(scores)
    print(f"\n  Summary: {positive}/{total} indicators positive")
    if positive >= 3:
        print("  ✓✓  FEASIBLE — cross-scale prediction is well-supported")
    elif positive >= 2:
        print("  ✓   POSSIBLE — weak but consistent signal; use weighted loss")
    elif positive >= 1:
        print("  △   MARGINAL — very weak signal; significant risk of failure")
    else:
        print("  ✗   NOT FEASIBLE — no detectable cross-scale predictive signal")
    print(f"{'═'*70}")


def _save_csv(all_results: list, path: str = "cross_scale_learnability_summary.csv"):
    import csv
    rows = []
    for r in all_results:
        row = {
            "target": r["label"],
            "mi_lag0_mean": r["mi"].get("mi_lag0_mean", ""),
            "mi_lag1_mean": r["mi"].get("mi_lag1_mean", ""),
            "mi_lag1_frac_pos": r["mi"].get("mi_lag1_frac_positive", ""),
            "lift_lag1": r["cond"].get("lift_lag1", ""),
            "p_long": r["cond"].get("p_long", ""),
            "peak_lag": r["lag"].get("peak_lag", ""),
            "peak_corr": r["lag"].get("peak_corr", ""),
        }
        for m in r["ml"].get("model_results", []):
            key = m["name"].replace(" ", "_").replace("(", "").replace(")", "")
            row[f"f1_{key}"] = m.get("f1", "")
            row[f"auc_{key}"] = m.get("auc", "")
        rows.append(row)

    if not rows:
        return

    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary CSV saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis for one target scale
# ─────────────────────────────────────────────────────────────────────────────

def analyse_cross_scale(
    target_w: int,
    pkldir: str,
    max_lag: int = 10,
    n_pairs_mi: int = 104196,   # all pairs by default
    n_pairs_ml: int = 3000,
    n_pairs_multi: int = 2000,
    history_lags: int = 4,
) -> dict:
    target_eps = TARGET_CONFIGS[target_w]
    label = f"w={target_w}, ε={target_eps}"

    print(f"\n{'#'*70}")
    print(f"  CROSS-SCALE LEARNABILITY: A_w35 (ε=0.1) → A_{target_w} (ε={target_eps})")
    print(f"{'#'*70}")

    # Load adjacency matrices
    print("\nLoading data …")
    src_daily = _load_adj(SOURCE_W, SOURCE_EPS, pkldir)     # (T, 457, 457)
    tgt_daily = _load_adj(target_w, target_eps, pkldir)     # (T, 457, 457)

    assert src_daily.shape[0] == tgt_daily.shape[0], (
        f"T mismatch: {src_daily.shape[0]} vs {tgt_daily.shape[0]}"
    )

    # Upper-triangle time series (daily)
    src_daily_up = _upper_tri_series(src_daily)   # (T, n_pairs)
    tgt_daily_up = _upper_tri_series(tgt_daily)   # (T, n_pairs)
    del src_daily, tgt_daily

    # Weekly snapshots
    src_weekly = src_daily_up[::WEEKLY_STRIDE]    # (T_w, n_pairs)
    tgt_weekly = tgt_daily_up[::WEEKLY_STRIDE]    # (T_w, n_pairs)
    del src_daily_up, tgt_daily_up

    T_w, n_pairs = src_weekly.shape
    print(f"\n  Weekly snapshots: {T_w}  |  Pairs: {n_pairs:,}")

    rng = np.random.default_rng(42)

    # ── Lens 1: Mutual Information ─────────────────────────────────────────
    mi_result = lens_mutual_information(src_weekly, tgt_weekly, label)

    # ── Lens 2: Conditional statistics ────────────────────────────────────
    cond_result = lens_conditional_stats(src_weekly, tgt_weekly, label)

    # ── Lens 3: Lead-lag correlation ───────────────────────────────────────
    lag_result = lens_leadlag(
        src_weekly, tgt_weekly, label,
        max_lag=max_lag,
        n_pairs_sample=5000,
        rng=rng,
    )

    # ── Lens 4: Baseline vs ML ─────────────────────────────────────────────
    ml_result = lens_ml_baseline(
        src_weekly, tgt_weekly, label,
        n_pairs_sample=n_pairs_ml,
        rng=rng,
    )

    # ── Lens 5: Multi-feature ML ───────────────────────────────────────────
    multi_result = lens_ml_multifeature(
        src_weekly, tgt_weekly, label,
        n_pairs_sample=n_pairs_multi,
        history_lags=history_lags,
        rng=rng,
    )

    # ── Verdict ────────────────────────────────────────────────────────────
    _print_verdict(label, mi_result, cond_result, lag_result, ml_result)

    return {
        "label": label,
        "target_w": target_w,
        "mi": mi_result,
        "cond": cond_result,
        "lag": lag_result,
        "ml": ml_result,
        "multi": multi_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Assess cross-scale learnability: can A_w35 (w=35, ε=0.1) predict\n"
            "A_w120 (w=120, ε=0.4) or A_w180 (w=180, ε=0.6)?\n\n"
            "Five diagnostic lenses:\n"
            "  1. Mutual Information\n"
            "  2. Conditional edge statistics / lift\n"
            "  3. Temporal lead-lag Pearson correlation\n"
            "  4. Baseline vs ML model (LogReg, RandomForest)\n"
            "  5. Multi-feature Gradient Boosting with temporal context\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run both target scales (w=120 and w=180).",
    )
    parser.add_argument(
        "--target", type=int, choices=[120, 180], default=None,
        help="Run a single target scale: 120 or 180.",
    )
    parser.add_argument(
        "--pkldir", type=str, default=None,
        help="Directory containing IQDw{w}.pkl files.",
    )
    parser.add_argument(
        "--max-lag", type=int, default=10,
        help="Maximum lag (weekly snapshots) for lead-lag analysis. Default=10.",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=3000,
        help="Number of pairs to subsample for ML lenses. Default=3000.",
    )
    parser.add_argument(
        "--history-lags", type=int, default=4,
        help="Number of historical lags for the multi-feature lens. Default=4.",
    )
    args = parser.parse_args()

    if not args.all and args.target is None:
        parser.error("Specify --all or --target {120|180}.")

    pkldir  = args.pkldir or DEFAULT_PKL_DIR
    targets = [120, 180] if args.all else [args.target]

    # Pre-check pkl files
    needed_ws = set([SOURCE_W] + targets)
    missing   = []
    for w in needed_ws:
        p = os.path.join(pkldir, f"IQDw{w}.pkl")
        if not os.path.isfile(p):
            missing.append(p)
    if missing:
        raise FileNotFoundError(
            "Missing pkl files:\n" + "\n".join(f"  {m}" for m in missing)
            + "\nUse --pkldir to specify the correct directory."
        )

    all_results = []
    lag_by_target: dict = {}

    for tw in targets:
        r = analyse_cross_scale(
            target_w=tw,
            pkldir=pkldir,
            max_lag=args.max_lag,
            n_pairs_ml=args.n_pairs,
            n_pairs_multi=max(args.n_pairs // 2, 500),
            history_lags=args.history_lags,
        )
        all_results.append(r)
        lag_by_target[tw] = r["lag"]

    # Lead-lag plot (combined if both targets run)
    plot_leadlag(
        results_120=lag_by_target.get(120),
        results_180=lag_by_target.get(180),
        save_path="cross_scale_leadlag.png",
    )

    # CSV summary
    _save_csv(all_results, "cross_scale_learnability_summary.csv")

    # Final cross-target comparison if both run
    if len(all_results) == 2:
        print(f"\n{'═'*70}")
        print("CROSS-TARGET COMPARISON SUMMARY")
        print(f"{'═'*70}")
        print(f"  {'Metric':<35}  {'→ w=120':>10}  {'→ w=180':>10}")
        print("  " + "─" * 58)
        for key, fmt in [
            ("mi.mi_lag1_mean",      "{:.5f}"),
            ("cond.lift_lag1",       "{:.2f}x"),
            ("lag.peak_lag",         "lag={}"),
            ("lag.peak_corr",        "{:.5f}"),
        ]:
            vals = []
            for r in all_results:
                k1, k2 = key.split(".")
                v = r[k1].get(k2, float("nan"))
                try:
                    vals.append(fmt.format(v))
                except Exception:
                    vals.append(str(v))
            print(f"  {key:<35}  {vals[0]:>10}  {vals[1]:>10}")
        print()
