"""
test_one_fold.py — Run all interpretability plots from a saved .pth checkpoint.

Usage (from the project root, next to transformer.py etc.):
    python test/test_one_fold.py

    # or pick a specific fold checkpoint:
    python test/test_one_fold.py --pth model_fold_3.pth

    # pass the distance-matrix pkl explicitly (required if cwd != project root):
    python test/test_one_fold.py --dm-pkl /path/to/IQDw35.pkl

    # run on a specific date-index in the distance matrix:
    python test/test_one_fold.py --sample-idx 100

    # skip data-dependent plots (gate values and LayerScale gammas only):
    python test/test_one_fold.py --no-data

Prerequisites
-------------
- At least one model_fold_N.pth file in the current working directory
  (or pass --pth explicitly).
- The distance-matrix pkl file accessible via the path in parameters.py
  (needed to build the input sample).  Pass --no-data if you only want
  gate/gamma plots and have no pkl available.

Model task
----------
The model predicts ΔD_t = D_{t+1} − D_t (the change in the distance matrix),
not the next level D_{t+1}.  All y tensors are scaled ΔD, using the fold-wise
y scaler (scaler_y_mean, scaler_y_std) saved in the checkpoint.
The null model predicts ΔD=0 (no change); in scaled space this is the constant
−scaler_y_mean / scaler_y_std.

Checkpoint format
-----------------
Each .pth file is a dict saved by train_with_validation():
    {
        "model_state_dict" : state_dict,
        "train_mse"        : list[float],
        "val_mse"          : list[float],
        "train_r2"         : list[float],   # R² vs zero-change null model
        "val_r2"           : list[float],
        "best_val_mse"     : float,         # val MSE at early-stopping epoch
        "best_epoch"       : int,
        "scaler_X_mean"    : float,
        "scaler_X_std"     : float,
        "scaler_y_mean"    : float,
        "scaler_y_std"     : float,
    }

What is produced
----------------
All plots are saved in ./interp_outputs/ (created automatically):

  fold_summary.png                      — CV summary (only when multiple folds found)
  attention_maps_block0.png             — effective attention maps, first block
  attention_maps_last_block.png         — effective attention maps, last block
  attention_maps_overlay_block0.png     — colour-coded overlay, first block
  attention_maps_overlay_last_block.png
  ind_attention_maps_overlay_block0.png
  ind_attention_maps_overlay_last_block.png
  gate_values.png                       — gate heatmap (no data needed)
  layerscale_gammas.png                 — LayerScale γ (no data needed)
  mean_attention_distance.png           — mean hop distance heatmap
  bar_mean_attention_distance.png       — per-block bar chart
  attention_weights.png                 — content-stream entropy violin plots
  prediction_error_map.png              — D_t / ΔD̂_t / ΔD_t / signed error / per-pixel skill score
                                          Panel 4 = ΔD̂−ΔD: red=over-predict, blue=under-predict
                                          Panel 5 = (ΔD̂−ΔD)²/ΔD²: green<1 beats null, red>1 worse, grey=|ΔD|≈0
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
from sklearn.model_selection import TimeSeriesSplit

# ── Make sure the project root is on sys.path so imports work whether
#    this file lives in test/ or in the root itself. ─────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "test" else _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from extract_distance_matrices import (
    extract_distance_matrix,
    reorder_by_gics,
    get_gics_sector_boundaries,
    build_patch_sector_ids,
)
from transformer import SmallDataDecoderViT
from model_interpretability import ModelInterpreter, plot_fold_summary


# ─────────────────────────────────────────────────────────────────────────────
# Model config — must match training_and_validation_functions.py exactly.
# If you change the architecture, update this dict and retrain.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CFG = dict(
    in_channels=1,
    embed_dim=96,        # must match main.py — reduced for better param/sample ratio
    depth=2,             # must match main.py
    num_heads=3,
    proj_drop=0.2,
    attn_drop=0.1,
    drop_path_rate=0.05,
    ls_init_value=1e-2,
    gate_init=0.0,       # must match main.py — sigmoid(0)=0.5 balanced init
)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_all_fold_checkpoints(root: str = ".") -> list[str]:
    """Return sorted list of model_fold_N.pth files found in root."""
    return sorted(glob.glob(os.path.join(root, "model_fold_*.pth")))


def _load_checkpoint(pth_path: str) -> dict:
    """
    Load a .pth file and return its contents as a normalised dict.

    Supports three formats:
      - New format (dict):  {"model_state_dict": ..., "train_mse": ..., ...}
      - New format with separate X/y scalers: also has scaler_X_mean, scaler_X_std, scaler_y_mean, scaler_y_std
      - Old format (bare state_dict):  {"pos_embed": ..., ...}
        Wrapped into the new schema with empty history lists so the rest of
        the code has a single interface to work with.
    """
    raw = torch.load(pth_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw  # new format — already correct

    # Old format: the file IS the state_dict.  Wrap it.
    return {
        "model_state_dict": raw,
        "train_mse": [],
        "val_mse":   [],
        "train_r2":  [],
        "val_r2":    [],
        "scaler_mean": None,
        "scaler_std":  None,
    }


def _load_fold_history(
    all_pths: list[str],
) -> tuple[list[dict], list[int]] | tuple[None, None]:
    """
    Read training history from each .pth file.

    Returns
    -------
    (histories, fold_numbers)
        histories    : list of fold-history dicts, one per checkpoint that
                       contains non-empty history.
        fold_numbers : corresponding 1-based fold indices parsed from filenames.
    Both are None if no checkpoint contains history (old-format files).
    """
    histories    = []
    fold_numbers = []

    for pth_path in all_pths:
        ckpt = _load_checkpoint(pth_path)
        if not ckpt["val_mse"]:
            # Old-format checkpoint — no history stored.
            continue
        m = re.search(r"model_fold_(\d+)\.pth$", os.path.basename(pth_path))
        fold_num = int(m.group(1)) if m else None
        histories.append(
            {k: ckpt.get(k) for k in ("train_mse", "val_mse", "train_r2", "val_r2", "best_val_mse")}
        )
        fold_numbers.append(fold_num)

    if not histories:
        return None, None
    return histories, fold_numbers


def _best_pth_from_history(
    all_pths: list[str],
    histories: list[dict],
    fold_numbers: list[int],
) -> str:
    """
    Pick the checkpoint with the lowest best_val_mse (the val MSE at the
    early-stopping checkpoint epoch), print a ranked table of all folds,
    and return the path to the best checkpoint.

    Falls back to final-epoch val_mse[-1] for old checkpoints that pre-date
    the best_val_mse field, with a warning.
    """
    final_val_mse = []
    for fh in histories:
        if fh.get("best_val_mse") is not None:
            final_val_mse.append(fh["best_val_mse"])
        else:
            # Old checkpoint format — best_val_mse not saved; use last epoch
            # (which is worse than the best, but it's the best we have).
            final_val_mse.append(fh["val_mse"][-1])

    best_idx      = int(np.argmin(final_val_mse))
    best_fold_num = fold_numbers[best_idx]

    print(f"\n── Val-MSE per fold (best checkpoint epoch)  [null-MSE ≈ 1.0 → skill = val-MSE / 1.0]:")
    print(f"  {'label':>8}  {'val-MSE':>10}  {'skill score':>12}  {'beats null?':>12}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*12}")
    for i, (fn, mse) in enumerate(zip(fold_numbers, final_val_mse)):
        marker      = " ← best" if i == best_idx else ""
        label       = f"fold {fn}" if fn is not None else f"entry {i+1}"
        skill       = mse          # null-MSE ≈ 1.0 by construction, so skill ≈ val-MSE
        beats_null  = "✓" if skill < 1.0 else "✗"
        print(f"  {label:>8}  {mse:>10.6f}  {skill:>12.6f}  {beats_null:>12}{marker}")

    # Pass 1: look for the expected filename in all_pths
    if best_fold_num is not None:
        candidate = f"model_fold_{best_fold_num}.pth"
        for pth in all_pths:
            if os.path.basename(pth) == candidate:
                return pth
        # File was in history but not on disk — warn and fall through
        print(
            f"  WARNING: expected checkpoint '{candidate}' not found on disk. "
            f"Falling back to last checkpoint."
        )

    # Pass 2: genuine fallback — return the last file in the sorted list
    return all_pths[-1]


def _load_model(pth_path: str, sector_ids: torch.Tensor) -> SmallDataDecoderViT:
    """
    Instantiate SmallDataDecoderViT using MODULE_CFG and load weights from
    a checkpoint.

    Handles both new-format dicts (key "model_state_dict") and old-format
    bare state_dicts, as well as the _orig_mod. prefix from torch.compile.
    """
    model = SmallDataDecoderViT(**MODEL_CFG, sector_ids=sector_ids)

    ckpt  = _load_checkpoint(pth_path)
    state = ckpt["model_state_dict"]

    # Strip _orig_mod. prefix if checkpoint was saved from a compiled model
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded weights from {pth_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_sector_ids_and_labels():
    """
    Build sector_ids tensor and auxiliary label structures without loading
    the full distance matrix.

    Uses a (N, N) dummy array so reorder_by_gics receives a 2-D input and
    only ticker/label outputs are needed (the reordered matrix is discarded).
    """
    dummy = np.zeros((457, 457), dtype=np.float32)
    _, tickers_gics, sector_labels = reorder_by_gics(dummy)
    sector_boundaries = get_gics_sector_boundaries(sector_labels)
    sector_ids        = build_patch_sector_ids(sector_labels)
    return sector_ids, sector_labels, sector_boundaries, tickers_gics


def _build_sample(
    distance_matrix_gics: np.ndarray,
    sample_idx: int,
    scaler_X_mean: float | None = None,
    scaler_X_std:  float | None = None,
    scaler_y_mean: float | None = None,
    scaler_y_std:  float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (X_sample, y_sample) tensors of shape (1, 1, 457, 457).

    X_sample : scaled D_t  (input to the model)
    y_sample : scaled ΔD_t = scaled (D_{t+1} − D_t)  (model target)

    sample_idx is clamped to [0, T-2].  The function walks backwards to
    avoid degenerate samples where D_t == D_{t+1} (forward-fill / holidays).

    Scalers must be the fold-wise train-set statistics from the checkpoint:
      X_sample = (D_t − scaler_X_mean) / scaler_X_std
      y_sample = ((D_{t+1} − D_t) − scaler_y_mean) / scaler_y_std
    If no scalers are provided the raw (unscaled) values are returned with
    a warning.
    """
    T   = distance_matrix_gics.shape[0]
    idx = max(0, min(sample_idx, T - 2))

    max_backtrack = 50
    backtracked   = 0
    while idx > 0 and backtracked < max_backtrack:
        if not np.array_equal(distance_matrix_gics[idx], distance_matrix_gics[idx + 1]):
            break
        idx        -= 1
        backtracked += 1

    if backtracked > 0:
        print(
            f"  NOTE: sample_idx had D_t == D_{{t+1}}; backtracked "
            f"{backtracked} step(s) to idx={idx} for a non-degenerate sample."
        )
    elif idx == 0 and T > 1 and np.array_equal(distance_matrix_gics[0], distance_matrix_gics[1]):
        print(
            "  WARNING: all consecutive time steps are identical. "
            "Using idx=0 despite D_0 == D_1."
        )

    D_t   = distance_matrix_gics[idx    ].astype(np.float32)
    D_tp1 = distance_matrix_gics[idx + 1].astype(np.float32)
    delta = D_tp1 - D_t   # ΔD_t

    if scaler_X_mean is not None and scaler_X_std is not None:
        D_t_scaled = (D_t - scaler_X_mean) / scaler_X_std
    else:
        print("  WARNING: no X scaler — returning unscaled D_t.")
        D_t_scaled = D_t

    if scaler_y_mean is not None and scaler_y_std is not None:
        delta_scaled = (delta - scaler_y_mean) / scaler_y_std
    else:
        print("  WARNING: no y scaler — returning unscaled ΔD_t.")
        delta_scaled = delta

    X = D_t_scaled[np.newaxis, np.newaxis, :]     # (1, 1, 457, 457)
    y = delta_scaled[np.newaxis, np.newaxis, :]   # (1, 1, 457, 457)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all interpretability plots from saved .pth checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pth",
        default=None,
        help=(
            "Path to a specific model_fold_N.pth file to evaluate.  "
            "When omitted, all model_fold_*.pth files in --root are loaded "
            "and the fold with the lowest final val-MSE is used."
        ),
    )
    parser.add_argument(
        "--root",
        default=".",
        help=(
            "Directory to search for model_fold_*.pth files "
            "(default: current working directory).  Ignored when --pth is set."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="interp_outputs",
        help="Directory where all plots are saved (default: interp_outputs/).",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=-1,
        help=(
            "Time index into the distance matrix to use as the input sample.  "
            "-1 (default) uses the last available time step (T-2)."
        ),
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help=(
            "Skip all plots that require the distance matrix.  "
            "Gate-value and LayerScale-gamma plots still run without it."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Transformer block index (0-based) for attention-map plots (default: 0).",
    )
    parser.add_argument(
        "--dm-pkl",
        default=None,
        dest="dm_pkl",
        help=(
            "Explicit path to the IQDw{w}.pkl distance-matrix file.  "
            "When omitted the path from extract_distance_matrices.py is used, "
            "which assumes the working directory is the project root."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1: sector metadata ───────────────────────────────────────────
    print("\n── Building sector metadata …")
    sector_ids, sector_labels, sector_boundaries, tickers_gics = (
        _build_sector_ids_and_labels()
    )
    print(f"  sector_ids shape : {tuple(sector_ids.shape)}")
    print(f"  unique sectors   : {sector_ids.unique().numel()}")

    # ── Step 2: pick checkpoint ───────────────────────────────────────────
    fold_history = None   # may remain None if --pth was given explicitly

    if args.pth:
        best_pth = args.pth
        if not os.path.isfile(best_pth):
            sys.exit(f"ERROR: checkpoint not found: {best_pth}")
        print(f"\n── Using specified checkpoint: {best_pth}")
    else:
        all_pths = _find_all_fold_checkpoints(args.root)
        if not all_pths:
            sys.exit(
                f"ERROR: no model_fold_*.pth files found in '{args.root}'.\n"
                "Either cd to the project root or pass --pth explicitly."
            )

        print(f"\n── Found {len(all_pths)} checkpoint(s) in '{args.root}':")
        for pth in all_pths:
            print(f"  {pth}")

        fold_history, fold_numbers = _load_fold_history(all_pths)

        if fold_history:
            best_pth = _best_pth_from_history(all_pths, fold_history, fold_numbers)
        else:
            best_pth = all_pths[-1]
            print(
                f"\n  NOTE: no training history found in checkpoints "
                f"(old-format .pth files?).\n"
                f"  Falling back to last checkpoint: {best_pth}"
            )

        print(f"\n  Selected checkpoint: {best_pth}")

    # ── Step 3: load model ────────────────────────────────────────────────
    print(f"\n── Loading model …")
    print(f"  Architecture: {MODEL_CFG}")
    model    = _load_model(best_pth, sector_ids)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"  Parameters   : {n_params:,}")

    interp     = ModelInterpreter(model, save_dir=args.out_dir)
    last_block = len(list(model.blocks)) - 1
    ckpt = _load_checkpoint(best_pth)
    # Get scalers - first try new separate X/y format, fall back to old single scaler
    scaler_X_mean = ckpt.get("scaler_X_mean")
    scaler_X_std  = ckpt.get("scaler_X_std")
    scaler_y_mean = ckpt.get("scaler_y_mean")
    scaler_y_std  = ckpt.get("scaler_y_std")
    scaler_mean   = ckpt.get("scaler_mean")
    scaler_std    = ckpt.get("scaler_std")

    # Print best-epoch skill score for this checkpoint (null-MSE ≈ 1.0)
    best_val_mse = ckpt.get("best_val_mse")
    best_epoch   = ckpt.get("best_epoch")
    if best_val_mse is not None:
        beats = "✓ beats null" if best_val_mse < 1.0 else "✗ does NOT beat null"
        print(f"  Best epoch   : {best_epoch}  |  "
              f"Best val MSE = {best_val_mse:.6f}  |  "
              f"Skill score ≈ {best_val_mse:.6f}  [{beats}]")

    # ── Step 4: fold summary (needs history) ─────────────────────────────
    if fold_history:
        print(f"\n── Plotting fold summary ({len(fold_history)} fold(s)) …")
        plot_fold_summary(
            fold_history,
            save_path=os.path.join(args.out_dir, "fold_summary.png"),
        )
    else:
        print(
            "\n  NOTE: no training history available — skipping fold summary.\n"
            "  (Use automatic checkpoint selection to generate fold_summary.png.)"
        )

    # ── Step 5: parameter-only plots (no distance matrix needed) ─────────
    print("\n── Gate values …")
    interp.plot_gate_values(filename="gate_values.png")

    print("\n── LayerScale gammas …")
    interp.plot_layerscale_gammas(filename="layerscale_gammas.png")

    if args.no_data:
        print("\n  --no-data set: skipping attention-map and error-map plots.")
        print(f"\nDone.  All outputs saved to '{args.out_dir}/'.")
        return

    # ── Step 6: load distance matrix ─────────────────────────────────────
    print("\n── Loading distance matrix …")
    try:
        distance_matrix_raw = extract_distance_matrix(pkl_path=args.dm_pkl)
    except FileNotFoundError as exc:
        print(
            f"\n  WARNING: could not load distance matrix:\n    {exc}\n"
            "  Skipping data-dependent plots.  "
            "Pass --no-data to suppress this warning, or --dm-pkl to fix the path."
        )
        print(f"\nDone (partial).  Outputs saved to '{args.out_dir}/'.")
        return

    print("  Reordering by GICS …")
    distance_matrix_gics, _, _ = reorder_by_gics(distance_matrix_raw)
    del distance_matrix_raw

    T          = distance_matrix_gics.shape[0]
    sample_idx = args.sample_idx if args.sample_idx >= 0 else T - 2
    print(f"  T={T} time steps.  Using sample index {sample_idx}.")

    # ── Build X (scaled D_t) and y (scaled ΔD_t) tensors ─────────────────
    # X and y use separate fold-wise scalers from the checkpoint.
    # Applying X scaler to D_t and y scaler to ΔD matches exactly what
    # diff_model_multi_fold_cv_train_test does during training.
    D      = distance_matrix_gics.astype(np.float32)   # (T, 457, 457)
    D_t    = D[:-1]                                     # (T-1, 457, 457)
    delta  = D[1:] - D[:-1]                             # ΔD_t, (T-1, 457, 457)

    if scaler_X_mean is not None and scaler_X_std is not None:
        print("  Applying fold-wise X scaler (D_t standardisation).")
        X_scaled = (D_t   - scaler_X_mean) / scaler_X_std
    elif scaler_mean is not None and scaler_std is not None:
        print("  Applying legacy single scaler (level prediction).")
        X_scaled = (D_t   - scaler_mean)   / scaler_std
    else:
        print("  NOTE: no X scaler in checkpoint — using raw D_t.")
        X_scaled = D_t

    if scaler_y_mean is not None and scaler_y_std is not None:
        print("  Applying fold-wise y scaler (ΔD standardisation).")
        y_scaled = (delta - scaler_y_mean) / scaler_y_std
    elif scaler_mean is not None and scaler_std is not None:
        # Legacy: single scaler was used for both X and next-level y
        print("  Applying legacy single scaler to ΔD (approximate).")
        y_scaled = (delta - scaler_mean)   / scaler_std
    else:
        print("  NOTE: no y scaler in checkpoint — using raw ΔD.")
        y_scaled = delta

    X_t = torch.from_numpy(X_scaled[:, np.newaxis, :]).float()  # (T-1, 1, 457, 457)
    y_t = torch.from_numpy(y_scaled[:, np.newaxis, :]).float()  # (T-1, 1, 457, 457)

    tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)
    *_, (_, last_val_idx) = tscv.split(X_t)
    X_val = X_t[last_val_idx]   # (N_val, 1, 457, 457)
    y_val = y_t[last_val_idx]   # (N_val, 1, 457, 457)

    # ── Null-model baseline (predict ΔD=0 in raw space) ──────────────────
    # In scaled space, ΔD=0 raw corresponds to −y_mean/y_std.
    if scaler_y_mean is not None and scaler_y_std is not None and float(scaler_y_std) != 0.0:
        null_value = float(-scaler_y_mean) / float(scaler_y_std)
    else:
        null_value = 0.0

    # Single-sample tensors
    sample_x, sample_y = _build_sample(
        distance_matrix_gics, sample_idx,
        scaler_X_mean=scaler_X_mean, scaler_X_std=scaler_X_std,
        scaler_y_mean=scaler_y_mean, scaler_y_std=scaler_y_std,
    )
    baseline_pred     = torch.full_like(sample_y, null_value)   # (1, 1, 457, 457)
    baseline_val_full = torch.full_like(y_val,    null_value)   # (N_val, 1, 457, 457)
    print(f"  sample_x shape: {tuple(sample_x.shape)}")

    # ── Step 7: data-dependent plots ─────────────────────────────────────
    print(f"\n── Attention maps (block {args.layer}) …")
    interp.plot_attention_maps(
        sample_x, layer=args.layer,
        filename=f"attention_maps_block{args.layer}.png",
    )

    print(f"\n── Attention maps (last block = {last_block}) …")
    interp.plot_attention_maps(
        sample_x, layer=last_block,
        filename="attention_maps_last_block.png",
    )

    print(f"\n── Attention overlay (block {args.layer}) …")
    interp.plot_attention_maps_overlay(
        sample_x, layer=args.layer,
        filename=f"attention_maps_overlay_block{args.layer}.png",
    )

    print(f"\n── Attention overlay (last block = {last_block}) …")
    interp.plot_attention_maps_overlay(
        sample_x, layer=last_block,
        filename="attention_maps_overlay_last_block.png",
    )

    print("\n── Mean attention distance …")
    interp.plot_mean_attention_distance(
        sample_x, filename="mean_attention_distance.png",
    )

    print("\n── Content attention weights …")
    interp.plot_attention_weights(sample_x, filename="attention_weights.png")

    print("\n── Prediction error map …")
    interp.plot_prediction_error_map(
        sample_x, sample_y,
        filename="prediction_error_map.png",
        tickers=tickers_gics,
        sector_boundaries=sector_boundaries,
        X_val=X_val,
        y_val=y_val,
        baseline_pred=baseline_pred,
        baseline_val_full=baseline_val_full,
        baseline_name="zero (ΔD=0)",
    )

    print(f"\n{'='*60}")
    print(f"Done.  All outputs saved to '{args.out_dir}/'.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
