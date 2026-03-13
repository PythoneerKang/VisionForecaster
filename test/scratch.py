"""
scratch.py — Run all interpretability plots from saved .pth checkpoints.

Usage (from the project root, next to transformer.py etc.):
    python test/scratch.py

    # or pick a specific fold checkpoint:
    python test/scratch.py --pth model_fold_3.pth

    # pass the distance-matrix pkl explicitly (required if cwd != project root):
    python test/scratch.py --dm-pkl /path/to/IQDw35.pkl

    # run on a specific date-index in the distance matrix:
    python test/scratch.py --sample-idx 100

    # skip data-dependent plots (gate values and LayerScale gammas only):
    python test/scratch.py --no-data

Prerequisites
-------------
- At least one model_fold_N.pth file in the current working directory
  (or pass --pth explicitly).
- The distance-matrix pkl file accessible via the path in parameters.py
  (needed to build the input sample).  If you only want gate/gamma plots
  that don't need data, pass --no-data.

Checkpoint format
-----------------
Each .pth file is a dict saved by train_with_validation():
    {
        "model_state_dict" : state_dict,
        "train_mse"        : list[float],
        "val_mse"          : list[float],
        "train_r2"         : list[float],
        "val_r2"           : list[float],
    }
Both weights and history live in the same file — no separate .pkl needed.

What is produced
----------------
All plots are saved in ./interp_outputs/ (created automatically):

  fold_summary.png                     — CV summary across all available folds
  attention_maps_block0.png            — effective attention maps, first block
  attention_maps_last_block.png        — effective attention maps, last block
  attention_maps_overlay_block0.png    — colour-coded overlay, first block
  attention_maps_overlay_last_block.png
  ind_attention_maps_overlay_block0.png
  ind_attention_maps_overlay_last_block.png
  gate_values.png                      — gate heatmap (no data needed)
  layerscale_gammas.png                — LayerScale γ (no data needed)
  mean_attention_distance.png          — mean hop distance heatmap
  bar_mean_attention_distance.png      — per-block bar chart
  attention_weights.png                — content-stream entropy violin plots
  prediction_error_map.png             — input / pred / truth / error panels
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

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
    embed_dim=192,
    depth=6,
    num_heads=3,
    proj_drop=0.1,
    drop_path_rate=0.05,
    ls_init_value=1e-2,
    gate_init=2.0,
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

    Supports two formats:
      - New format (dict):  {"model_state_dict": ..., "train_mse": ..., ...}
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
            {k: ckpt[k] for k in ("train_mse", "val_mse", "train_r2", "val_r2")}
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
    Pick the checkpoint with the lowest final-epoch val-MSE, print a ranked
    table of all folds, and return the path to the best checkpoint.

    The search is done in two passes:
      1. Try to find model_fold_{best_fold_num}.pth in all_pths (reliable).
      2. Fall back to all_pths[-1] only if the file genuinely cannot be found,
         printing a clear warning so the user knows something went wrong.
    """
    final_val_mse = [fh["val_mse"][-1] for fh in histories]
    best_idx      = int(np.argmin(final_val_mse))
    best_fold_num = fold_numbers[best_idx]

    print(f"\n── Val-MSE per fold (final epoch):")
    for i, (fn, mse) in enumerate(zip(fold_numbers, final_val_mse)):
        marker = " ← best" if i == best_idx else ""
        label  = f"fold {fn}" if fn is not None else f"entry {i+1}"
        print(f"  {label:>8}  val-MSE = {mse:.6f}{marker}")

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (X_sample, y_sample) tensors of shape (1, 1, 457, 457).
    sample_idx is clamped to [0, T-2].

    Walks backwards from sample_idx to avoid degenerate consecutive-duplicate
    samples where x_t == y_{t+1} exactly (e.g. from forward-fill or market
    holidays), since those make the naive baseline MSE = 0.
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
            f"  NOTE: sample_idx had x_t == y_t+1 exactly; backtracked "
            f"{backtracked} step(s) to idx={idx} for a non-degenerate sample."
        )

    X = distance_matrix_gics[idx    ][np.newaxis, np.newaxis, :]   # (1,1,457,457)
    y = distance_matrix_gics[idx + 1][np.newaxis, np.newaxis, :]
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

    sample_x, sample_y = _build_sample(distance_matrix_gics, sample_idx)
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
    )

    print(f"\n{'='*60}")
    print(f"Done.  All outputs saved to '{args.out_dir}/'.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
