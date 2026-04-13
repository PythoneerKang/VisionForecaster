"""
check_flip_rate.py
==================
Diagnostic script to measure how much the adjacency matrix actually
changes from day to day, across the four (w, epsilon) research combos.

The four research combinations are:
    w=35,  epsilon=0.1
    w=70,  epsilon=0.2
    w=120, epsilon=0.4
    w=180, epsilon=0.6

Usage
-----
# Check all four combos in one run (recommended):
    python check_flip_rate.py --all

# Check a single combo by index (0-based):
    python check_flip_rate.py --combo 0    # w=35,  eps=0.1
    python check_flip_rate.py --combo 1    # w=70,  eps=0.2
    python check_flip_rate.py --combo 2    # w=120, eps=0.4
    python check_flip_rate.py --combo 3    # w=180, eps=0.6

# Override the pkl base directory (if your files are elsewhere):
    python check_flip_rate.py --all --pkldir /path/to/pkl/folder

Each combo loads its own IQDw{w}.pkl file, since w controls the
correlation window width and each w has its own precomputed pkl.
"""

import argparse
import os
import pickle
import numpy as np
import parameters as p


# ─────────────────────────────────────────────────────────────────────────────
# Research combos  (w, epsilon)
# ─────────────────────────────────────────────────────────────────────────────

RESEARCH_COMBOS = [
    (35,  0.1),
    (70,  0.2),
    (120, 0.4),
    (180, 0.6),
]


# ─────────────────────────────────────────────────────────────────────────────
# Default pkl base directory  (mirrors extract_distance_matrices.py)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PKL_DIR = (
    "../Quasi_Differentiation_High_Temporal_Resolution_Cross_Correlations/Codes/"
    "Extract distance matrix (2017-2022) from pkl file"
)


def pkl_path_for(w: int, pkldir: str) -> str:
    return os.path.join(pkldir, f"IQDw{w}.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Load & convert distance matrix
# ─────────────────────────────────────────────────────────────────────────────

def load_distance(w: int, pkldir: str) -> np.ndarray:
    """
    Load IQDw{w}.pkl and return the (T, 457, 457) float32 distance matrix.
    Applies the correlation -> distance conversion and removes 6 bad tickers.
    """
    path = pkl_path_for(w, pkldir)
    print(f"  Loading: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f).astype(np.float32)

    # correlation -> distance:  d = sqrt(2 * (1 - corr))
    data = np.clip(data, -1.0, 1.0)
    dist = (2.0 * (1.0 - data)) ** np.float32(0.5)
    del data

    # remove 6 bad tickers: ABMD(6), CTVA(111), DOW(128), FOX(169), FOXA(170), IR(225)
    bad  = [6, 111, 128, 169, 170, 225]
    dist = np.delete(dist, bad, axis=1)
    dist = np.delete(dist, bad, axis=2)

    assert dist.shape[1:] == (457, 457), (
        f"Unexpected shape {dist.shape} after removing bad tickers."
    )
    return dist


def distance_to_adjacency(dist: np.ndarray, epsilon: float) -> np.ndarray:
    """Convert (T, N, N) float distance matrix to binary {0,1} adjacency."""
    N   = dist.shape[1]
    adj = dist.copy()
    adj[adj <= epsilon] = 1
    adj[adj != 1]       = 0
    adj[:, np.arange(N), np.arange(N)] = 0   # zero self-loops
    return adj.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis for one (w, epsilon) combo
# ─────────────────────────────────────────────────────────────────────────────

def analyse(w: int, epsilon: float, pkldir: str) -> dict:
    """
    Load IQDw{w}.pkl, convert to adjacency at given epsilon,
    compute flip-rate statistics, print a report, and return a summary dict.
    """
    print(f"\n{'='*62}")
    print(f"COMBO  w={w}, epsilon={epsilon}")
    print(f"{'='*62}")

    dist     = load_distance(w, pkldir)
    adj      = distance_to_adjacency(dist, epsilon)
    del dist

    T, N, _  = adj.shape
    n_pairs  = N * (N - 1) // 2

    print(f"  Shape : {adj.shape}  (T={T} days, N={N} stocks)")
    print(f"  Pairs : {n_pairs:,}  (upper triangle, no diagonal)")

    # Upper triangle only — adjacency is symmetric, diagonal is always 0
    triu      = np.triu_indices(N, k=1)
    adj_upper = adj[:, triu[0], triu[1]]           # (T, n_pairs)  binary {0,1}
    del adj

    # ── Edge density ──────────────────────────────────────────────────────
    density_per_day = adj_upper.mean(axis=1)       # fraction of pairs with edge

    print(f"\n  EDGE DENSITY  (fraction of pairs with an active edge)")
    print(f"    Mean   : {density_per_day.mean():.4f}  "
          f"({density_per_day.mean() * n_pairs:,.0f} edges/day)")
    print(f"    Std    : {density_per_day.std():.4f}")
    print(f"    Min    : {density_per_day.min():.4f}  (day {density_per_day.argmin()})")
    print(f"    Max    : {density_per_day.max():.4f}  (day {density_per_day.argmax()})")

    # ── Daily flip rate ───────────────────────────────────────────────────
    delta_abs         = np.abs(np.diff(adj_upper, axis=0))  # (T-1, n_pairs) {0,1}
    flip_rate_per_day = delta_abs.mean(axis=1)              # fraction flipped
    flips_per_day     = delta_abs.sum(axis=1)               # absolute count

    delta_signed = np.diff(adj_upper, axis=0)               # {-1, 0, +1}
    appear_rate  = (delta_signed ==  1).mean()              # 0->1
    vanish_rate  = (delta_signed == -1).mean()              # 1->0
    stable_rate  = (delta_signed ==  0).mean()              # unchanged

    fr = flip_rate_per_day.mean()

    print(f"\n  DAILY FLIP RATE  (fraction of pairs that change state)")
    print(f"    Mean   : {fr:.4f}  ({flips_per_day.mean():.0f} pairs/day)")
    print(f"    Std    : {flip_rate_per_day.std():.4f}")
    print(f"    Median : {np.median(flip_rate_per_day):.4f}")
    print(f"    Min    : {flip_rate_per_day.min():.4f}  "
          f"({flips_per_day.min():.0f} pairs, day {flip_rate_per_day.argmin()})")
    print(f"    Max    : {flip_rate_per_day.max():.4f}  "
          f"({flips_per_day.max():.0f} pairs, day {flip_rate_per_day.argmax()})")

    print(f"\n  FLIP BREAKDOWN  (averaged over all days and pairs)")
    print(f"    Stable (no change) : {stable_rate*100:.3f}%")
    print(f"    Appear (0 -> 1)    : {appear_rate*100:.3f}%")
    print(f"    Vanish (1 -> 0)    : {vanish_rate*100:.3f}%")
    print(f"    Total flips        : {(appear_rate+vanish_rate)*100:.3f}%")

    print(f"\n  PERSISTENCE BASELINE MSE : {fr:.6f}")
    print(f"  (= mean flip rate; each flip costs exactly 1^2 under binary MSE)")

    # ── Verdict ───────────────────────────────────────────────────────────
    if fr < 0.02:
        verdict = "VERY LOW  (<2%)  -- almost certainly unlearnable with plain MSE"
    elif fr < 0.08:
        verdict = "LOW  (2-8%)  -- weighted loss recommended"
    elif fr < 0.20:
        verdict = "MODERATE  (8-20%)  -- MSE training should be tractable"
    else:
        verdict = "HIGH  (>20%)  -- persistence is a weak baseline, good signal"

    print(f"\n  VERDICT: {verdict}")

    return {
        "w":                  w,
        "epsilon":            epsilon,
        "T":                  T,
        "mean_density":       float(density_per_day.mean()),
        "mean_edges_per_day": float(density_per_day.mean() * n_pairs),
        "mean_flip_rate":     float(fr),
        "median_flip_rate":   float(np.median(flip_rate_per_day)),
        "mean_flips_per_day": float(flips_per_day.mean()),
        "appear_pct":         float(appear_rate * 100),
        "vanish_pct":         float(vanish_rate * 100),
        "persistence_mse":    float(fr),
        "verdict":            verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-combo summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list) -> None:
    print(f"\n\n{'='*76}")
    print(f"CROSS-COMBO SUMMARY TABLE")
    print(f"{'='*76}")
    print(
        f"  {'w':>5}  {'eps':>5}  {'T':>5}  {'density%':>9}  "
        f"{'flip%':>7}  {'flips/day':>10}  {'persist_MSE':>12}"
    )
    print("  " + "-" * 72)
    for r in results:
        print(
            f"  {r['w']:>5}  {r['epsilon']:.1f}   "
            f"{r['T']:>5}  "
            f"{r['mean_density']*100:>8.3f}%  "
            f"{r['mean_flip_rate']*100:>6.3f}%  "
            f"{r['mean_flips_per_day']:>10.0f}  "
            f"{r['persistence_mse']:>12.6f}"
        )
    print()

    print("TRACTABILITY GUIDE:")
    print("  flip% < 2%    very hard with plain MSE  (model collapses to persistence)")
    print("  flip% 2-8%    hard     -- weighted loss recommended")
    print("  flip% 8-20%   tractable with plain MSE")
    print("  flip% > 20%   easy     -- persistence is a weak baseline")

    print("\nCROSS-SCALE TRANSFER NOTE:")
    print("  M(w_low, eps_low) -> predict A(w_high, eps_high):")
    print("  The source model must have learned something non-trivial")
    print("  (flip% > 2%) for transfer to be meaningful.")
    print("  If M(35, 0.1) collapses to persistence, its transferred")
    print("  weights carry no learnable signal to the next scale.")
    print(f"\n{'='*76}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Check adjacency flip rate for the four (w, epsilon) research combos.\n"
            "Combos: (35,0.1)  (70,0.2)  (120,0.4)  (180,0.6)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all four research combos: (35,0.1) (70,0.2) (120,0.4) (180,0.6).",
    )
    parser.add_argument(
        "--combo", type=int, default=None, choices=[0, 1, 2, 3],
        help=(
            "Run a single combo by index:  "
            "0=(w=35,eps=0.1)  1=(w=70,eps=0.2)  "
            "2=(w=120,eps=0.4)  3=(w=180,eps=0.6)."
        ),
    )
    parser.add_argument(
        "--pkldir", type=str, default=None,
        help=(
            "Directory containing the IQDw{w}.pkl files. "
            "Defaults to the path in extract_distance_matrices.py."
        ),
    )
    args = parser.parse_args()

    if not args.all and args.combo is None:
        parser.error("Specify --all to run all combos, or --combo N for a single one.")

    pkldir = args.pkldir if args.pkldir is not None else DEFAULT_PKL_DIR

    # Determine which combos to run
    if args.all:
        combos = RESEARCH_COMBOS
    else:
        combos = [RESEARCH_COMBOS[args.combo]]

    # Check all pkl files exist before starting
    missing = []
    for w, _ in combos:
        path = pkl_path_for(w, pkldir)
        if not os.path.isfile(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "The following pkl files were not found:\n"
            + "\n".join(f"  {m}" for m in missing)
            + f"\nPass the correct directory with --pkldir /path/to/folder"
        )

    # Run analysis for each combo
    results = []
    for w, epsilon in combos:
        r = analyse(w, epsilon, pkldir)
        results.append(r)

    # Print comparison table if more than one combo was run
    if len(results) > 1:
        print_summary_table(results)
