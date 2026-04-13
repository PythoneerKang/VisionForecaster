"""
check_weekly_flip_rate.py
=========================
Diagnose whether weekly-aggregated filtered distance matrices have enough
edge-state variability for a deep learning model to learn from.

For weekly forecasting the question is NOT "how often does an edge flip
day-to-day?" but rather "how often does the weekly snapshot differ from
the previous weekly snapshot?"  These are very different numbers:

    daily flip rate   : fraction of edges that change on any given day
    weekly flip rate  : fraction of edges that differ between consecutive
                        Friday-to-Friday (or every-5th-day) snapshots

Because many flips within a week can cancel (0→1→0 counts as 0 net change),
the weekly flip rate is generally LOWER than 5× the daily rate.

The four research combos tested:
    w=35,  epsilon=0.1
    w=70,  epsilon=0.2
    w=120, epsilon=0.4
    w=180, epsilon=0.6

Usage
-----
    python check_weekly_flip_rate.py --all
    python check_weekly_flip_rate.py --combo 0   # w=35, eps=0.1
    python check_weekly_flip_rate.py --all --stride 5   # 5-day (weekly) stride (default)
    python check_weekly_flip_rate.py --all --stride 1   # reproduce daily for comparison
    python check_weekly_flip_rate.py --all --pkldir /path/to/pkls
"""

import argparse
import os
import pickle
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RESEARCH_COMBOS = [
    (35,  0.1),
    (70,  0.2),
    (120, 0.4),
    (180, 0.6),
]

DEFAULT_PKL_DIR = (
    "../Quasi_Differentiation_High_Temporal_Resolution_Cross_Correlations/Codes/"
    "Extract distance matrix (2017-2022) from pkl file"
)

BAD_INDICES = [6, 111, 128, 169, 170, 225]   # ABMD, CTVA, DOW, FOX, FOXA, IR


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_adjacency(w: int, epsilon: float, pkldir: str) -> np.ndarray:
    """
    Load IQDw{w}.pkl → (T, 457, 457) binary float32 adjacency matrix.
    Correlation → distance → threshold at epsilon → remove 6 bad tickers.
    """
    path = os.path.join(pkldir, f"IQDw{w}.pkl")
    print(f"  Loading: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f).astype(np.float32)

    data = np.clip(data, -1.0, 1.0)
    dist = np.sqrt(2.0 * (1.0 - data)).astype(np.float32)
    del data

    adj = dist.copy()
    adj[adj <= epsilon] = 1.0
    adj[adj != 1.0]     = 0.0
    adj[:, np.arange(dist.shape[1]), np.arange(dist.shape[1])] = 0.0
    del dist

    adj = np.delete(adj, BAD_INDICES, axis=1)
    adj = np.delete(adj, BAD_INDICES, axis=2)
    assert adj.shape[1:] == (457, 457)
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse(w: int, epsilon: float, pkldir: str, stride: int) -> dict:
    """
    Full diagnostic for one (w, epsilon) combo at a given temporal stride.

    stride=1  → daily snapshots (matches check_flip_rate.py baseline)
    stride=5  → weekly snapshots (every 5th trading day)
    """
    freq_label = "DAILY" if stride == 1 else f"{stride}-DAY (WEEKLY)" if stride == 5 else f"{stride}-DAY"

    print(f"\n{'='*66}")
    print(f"COMBO  w={w}, epsilon={epsilon}  |  stride={stride}  [{freq_label}]")
    print(f"{'='*66}")

    adj = load_adjacency(w, epsilon, pkldir)   # (T, 457, 457)
    T, N, _ = adj.shape
    n_pairs = N * (N - 1) // 2

    # ── Sub-sample to the desired stride ─────────────────────────────────
    # Take every `stride`-th day starting from index 0.
    # For weekly: days 0, 5, 10, … ≈ Monday of each week (if data starts Monday)
    # This gives T_w snapshots and T_w-1 consecutive pairs to compare.
    adj_strided = adj[::stride]                 # (T_w, 457, 457)
    T_w = len(adj_strided)

    print(f"  Full daily shape  : {adj.shape}   (T={T} trading days)")
    print(f"  Strided shape     : {adj_strided.shape}  "
          f"(T_w={T_w} snapshots, stride={stride})")
    print(f"  Pairs (upper tri) : {n_pairs:,}")
    del adj

    # Upper triangle only
    triu = np.triu_indices(N, k=1)
    snap = adj_strided[:, triu[0], triu[1]]     # (T_w, n_pairs) binary {0,1}
    del adj_strided

    # ── Edge density ──────────────────────────────────────────────────────
    density = snap.mean(axis=1)                 # fraction of edges active
    mean_edges = density.mean() * n_pairs

    print(f"\n  EDGE DENSITY  (fraction of upper-tri pairs with active edge)")
    print(f"    Mean   : {density.mean():.5f}  ({mean_edges:.0f} edges / snapshot)")
    print(f"    Std    : {density.std():.5f}")
    print(f"    Min    : {density.min():.5f}  (snapshot {density.argmin()})")
    print(f"    Max    : {density.max():.5f}  (snapshot {density.argmax()})")

    # ── Flip rate between consecutive strided snapshots ───────────────────
    delta          = np.diff(snap.astype(np.int8), axis=0)  # {-1, 0, +1}
    flip_abs       = np.abs(delta)                           # {0, 1}
    flip_rate_snap = flip_abs.mean(axis=1)                   # per snapshot
    flips_snap     = flip_abs.sum(axis=1)

    appear_rate = (delta ==  1).mean()   # 0→1
    vanish_rate = (delta == -1).mean()   # 1→0
    stable_rate = (delta ==  0).mean()

    fr      = float(flip_rate_snap.mean())
    fr_med  = float(np.median(flip_rate_snap))

    print(f"\n  FLIP RATE  (fraction of pairs that change between consecutive snapshots)")
    print(f"    Mean   : {fr:.5f}  ({flips_snap.mean():.0f} pair-flips / transition)")
    print(f"    Std    : {flip_rate_snap.std():.5f}")
    print(f"    Median : {fr_med:.5f}")
    print(f"    Min    : {flip_rate_snap.min():.5f}  "
          f"({flips_snap.min():.0f} flips, snapshot {flip_rate_snap.argmin()})")
    print(f"    Max    : {flip_rate_snap.max():.5f}  "
          f"({flips_snap.max():.0f} flips, snapshot {flip_rate_snap.argmax()})")

    print(f"\n  FLIP BREAKDOWN  (averaged over all transitions and pairs)")
    print(f"    Stable (no change) : {stable_rate*100:.4f}%")
    print(f"    Appear (0 → 1)     : {appear_rate*100:.4f}%")
    print(f"    Vanish (1 → 0)     : {vanish_rate*100:.4f}%")
    print(f"    Total flips        : {(appear_rate+vanish_rate)*100:.4f}%")

    # ── Persistence baseline (key DL learnability metric) ─────────────────
    # Under binary MSE, if a model always predicts "no change", its error equals
    # the flip rate.  A DL model must beat this to be useful.
    print(f"\n  PERSISTENCE BASELINE MSE  : {fr:.8f}")
    print(f"  (= mean flip rate; predicting no-change costs exactly this under binary MSE)")

    # ── Auto-correlation of edge density (regime persistence) ─────────────
    # High autocorrelation → edges are regime-like → easier to predict
    if T_w > 10:
        density_centered = density - density.mean()
        var = (density_centered**2).mean()
        if var > 0:
            lag1_ac = float((density_centered[:-1] * density_centered[1:]).mean() / var)
        else:
            lag1_ac = 0.0
        print(f"\n  EDGE DENSITY LAG-1 AUTOCORRELATION : {lag1_ac:.4f}")
        print(f"  (closer to 1.0 → density is regime-like and predictable)")
    else:
        lag1_ac = float("nan")

    # ── Distribution of per-edge flip counts ─────────────────────────────
    # How many pairs are never/rarely vs often flipping?
    # Pairs that flip frequently are the ones the model can potentially learn.
    pair_flip_counts = flip_abs.sum(axis=0)    # (n_pairs,) total flips over time
    max_flips = T_w - 1                         # maximum possible flips

    never_flip_frac   = float((pair_flip_counts == 0).mean())
    low_flip_frac     = float((pair_flip_counts <= max_flips * 0.05).mean())  # ≤5% of time
    high_flip_frac    = float((pair_flip_counts >= max_flips * 0.20).mean())  # ≥20% of time

    print(f"\n  PER-PAIR FLIP FREQUENCY DISTRIBUTION")
    print(f"    Never flip (0 flips)        : {never_flip_frac*100:.2f}% of pairs")
    print(f"    Rarely flip (≤5% of time)   : {low_flip_frac*100:.2f}% of pairs")
    print(f"    Frequently flip (≥20% time) : {high_flip_frac*100:.2f}% of pairs")
    print(f"    Mean flips per pair         : {pair_flip_counts.mean():.2f} / {max_flips}")

    # ── Class imbalance ───────────────────────────────────────────────────
    # flip_abs values are the "positive class" for prediction.
    # Extreme imbalance → need focal/weighted loss.
    total_cells = flip_abs.size
    total_flips = flip_abs.sum()
    imbalance_ratio = int(total_cells / max(total_flips, 1))

    print(f"\n  CLASS IMBALANCE (flip=1 is the minority class)")
    print(f"    Total cells              : {total_cells:,}")
    print(f"    Cells with flip (1)      : {int(total_flips):,}  ({fr*100:.4f}%)")
    print(f"    Cells without flip (0)   : {total_cells - int(total_flips):,}  ({(1-fr)*100:.4f}%)")
    print(f"    Imbalance ratio (0:1)    : {imbalance_ratio}:1")
    if imbalance_ratio > 50:
        print(f"    ⚠  SEVERE imbalance — use weighted or focal loss for binary prediction")
    elif imbalance_ratio > 10:
        print(f"    ⚠  MODERATE imbalance — weighted loss recommended")
    else:
        print(f"    ✓  Manageable imbalance — standard loss should work")

    # ── Learnability verdict ──────────────────────────────────────────────
    print(f"\n  {'─'*58}")
    if fr < 0.005:
        verdict = "ESSENTIALLY ZERO  (<0.5%) — model will collapse to all-zeros; unlearnable"
        symbol  = "✗✗"
    elif fr < 0.02:
        verdict = "VERY LOW  (0.5–2%) — near-unlearnable with plain MSE; use weighted/focal loss"
        symbol  = "✗"
    elif fr < 0.05:
        verdict = "LOW  (2–5%) — hard but feasible with weighted loss and careful training"
        symbol  = "△"
    elif fr < 0.15:
        verdict = "MODERATE  (5–15%) — learnable; standard training should work"
        symbol  = "✓"
    else:
        verdict = "HIGH  (>15%) — persistence is a weak baseline; ample signal"
        symbol  = "✓✓"

    print(f"  {symbol}  LEARNABILITY: {verdict}")
    print(f"  {'─'*58}")

    return {
        "w": w, "epsilon": epsilon, "stride": stride,
        "T": T, "T_w": T_w, "n_pairs": n_pairs,
        "mean_density": float(density.mean()),
        "mean_edges": float(mean_edges),
        "mean_flip_rate": fr,
        "median_flip_rate": fr_med,
        "mean_flips_per_snap": float(flips_snap.mean()),
        "appear_pct": float(appear_rate * 100),
        "vanish_pct": float(vanish_rate * 100),
        "lag1_ac": lag1_ac,
        "never_flip_frac": never_flip_frac,
        "imbalance_ratio": imbalance_ratio,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-combo summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list, stride: int) -> None:
    freq = "WEEKLY" if stride == 5 else f"STRIDE-{stride}"
    print(f"\n\n{'='*80}")
    print(f"CROSS-COMBO SUMMARY  [{freq}]")
    print(f"{'='*80}")
    print(
        f"  {'w':>5}  {'eps':>5}  {'T_w':>5}  {'density%':>9}  "
        f"{'flip%':>7}  {'flips/snap':>11}  {'lag1_AC':>8}  {'imbal':>7}"
    )
    print("  " + "─" * 76)
    for r in results:
        print(
            f"  {r['w']:>5}  {r['epsilon']:.1f}   "
            f"{r['T_w']:>5}  "
            f"{r['mean_density']*100:>8.3f}%  "
            f"{r['mean_flip_rate']*100:>6.3f}%  "
            f"{r['mean_flips_per_snap']:>11.0f}  "
            f"{r['lag1_ac']:>8.4f}  "
            f"{r['imbalance_ratio']:>5}:1"
        )

    print(f"\n  LEARNABILITY GUIDE ({freq}):")
    print("    flip% < 0.5%    ✗✗  essentially unlearnable (all-zero collapse)")
    print("    flip% 0.5–2%   ✗   very hard — needs weighted/focal loss")
    print("    flip% 2–5%     △   hard but feasible with weighted loss")
    print("    flip% 5–15%    ✓   learnable with standard training")
    print("    flip% > 15%    ✓✓  easy — persistence is a weak baseline")

    print(f"\n  LAG-1 AUTOCORRELATION NOTE:")
    print("    AC > 0.9 → density is highly regime-like → temporal context useful")
    print("    AC < 0.5 → density is noisy → harder to exploit temporal patterns")

    print(f"\n  IMBALANCE NOTE:")
    print("    >50:1 → focal loss or heavy class-weighting strongly recommended")
    print("    10–50:1 → weighted BCE or weighted MSE recommended")
    print("    <10:1 → standard loss likely fine")
    print(f"\n{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weekly flip-rate diagnostic for filtered distance matrices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all four (w, epsilon) combos.")
    parser.add_argument("--combo", type=int, default=None, choices=[0, 1, 2, 3],
                        help="Single combo index: 0=(35,0.1) 1=(70,0.2) "
                             "2=(120,0.4) 3=(180,0.6)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Snapshot stride in trading days. "
                             "5 = weekly (default). 1 = daily (for comparison).")
    parser.add_argument("--pkldir", type=str, default=None,
                        help="Directory containing IQDw{w}.pkl files.")
    args = parser.parse_args()

    if not args.all and args.combo is None:
        parser.error("Specify --all or --combo N.")

    pkldir = args.pkldir or DEFAULT_PKL_DIR
    combos = RESEARCH_COMBOS if args.all else [RESEARCH_COMBOS[args.combo]]

    # Pre-check all pkl files
    missing = [
        os.path.join(pkldir, f"IQDw{w}.pkl")
        for w, _ in combos
        if not os.path.isfile(os.path.join(pkldir, f"IQDw{w}.pkl"))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing pkl files:\n" + "\n".join(f"  {m}" for m in missing)
            + f"\nUse --pkldir to specify the correct directory."
        )

    results = []
    for w, epsilon in combos:
        r = analyse(w, epsilon, pkldir, stride=args.stride)
        results.append(r)

    if len(results) > 1:
        print_summary(results, stride=args.stride)

    # ── Quick daily vs weekly comparison for the active combo(s) ─────────
    if args.stride != 1:
        print(f"\n{'='*66}")
        print(f"DAILY vs WEEKLY FLIP RATE COMPARISON")
        print(f"{'='*66}")
        print(f"  {'w':>5}  {'eps':>5}  {'daily_flip%':>12}  {'weekly_flip%':>13}  {'ratio':>7}")
        print("  " + "─" * 50)
        for r in results:
            r_daily = analyse(r['w'], r['epsilon'], pkldir, stride=1)
            ratio = r['mean_flip_rate'] / max(r_daily['mean_flip_rate'], 1e-12)
            print(
                f"  {r['w']:>5}  {r['epsilon']:.1f}   "
                f"{r_daily['mean_flip_rate']*100:>11.4f}%  "
                f"{r['mean_flip_rate']*100:>12.4f}%  "
                f"{ratio:>6.3f}x"
            )
        print(f"\n  ratio < 5× indicates many within-week flips cancel out")
        print(f"  (0→1→0 within a week = 0 net change = 0 weekly flip)\n")
