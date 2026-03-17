"""
test_one_fold.py — Run all interpretability plots from a saved .pth checkpoint.

UPDATED: Now works with any model configuration (original, optimized, or improved)
         Auto-detects model architecture from checkpoint or uses CLI args.

Usage (from the project root, next to transformer.py etc.):
    python test/test_one_fold.py

    # or pick a specific fold checkpoint:
    python test/test_one_fold.py --pth best_model_fold_3.pt

    # pass the distance-matrix pkl explicitly (required if cwd != project root):
    python test/test_one_fold.py --dm-pkl /path/to/IQDw35.pkl

    # run on a specific date-index in the distance matrix:
    python test/test_one_fold.py --sample-idx 100

    # skip data-dependent plots (gate values and LayerScale gammas only):
    python test/test_one_fold.py --no-data
    
    # override model configuration (if not auto-detected from checkpoint):
    python test/test_one_fold.py --embed-dim 48 --depth 1 --patch-size 24

Prerequisites
-------------
- At least one .pt file in the current working directory
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
Each .pt file is a dict saved by train_with_validation_optimized():
    {
        "model_state_dict"    : state_dict,
        "optimizer_state_dict": state_dict,
        "epoch"               : int,          # epoch at which this checkpoint was saved
        "val_r2"              : float,        # best val R² scalar at this checkpoint
        "scaler_X_mean"       : float,
        "scaler_X_std"        : float,
        "scaler_y_mean"       : float,
        "scaler_y_std"        : float,
        "train_loss"          : list[float],  # per-epoch mean MSE (train)
        "val_loss"            : list[float],  # per-epoch mean MSE (val)
        "train_r2"            : list[float],  # per-epoch R² vs zero-change null (train)
        "val_r2_history"      : list[float],  # per-epoch R² vs zero-change null (val)
                                              # named val_r2_history to avoid collision
                                              # with the scalar val_r2 above
        "train_mae"           : list[float],
        "val_mae"             : list[float],
    }

What is produced
----------------
All plots are saved in ./interp_outputs/ (created automatically):

  fold_summary.png                      — CV summary (only when multiple folds found)
  attention_maps_block0.png             — effective attention maps, first block
  attention_maps_last_block.png         — effective attention maps, last block (if depth>1)
  attention_maps_overlay_block0.png     — colour-coded overlay, first block
  attention_maps_overlay_last_block.png — (if depth>1)
  gate_values.png                       — gate heatmap (no data needed)
  layerscale_gammas.png                 — LayerScale γ (no data needed)
  mean_attention_distance.png           — mean hop distance heatmap
  attention_weights.png                 — content-stream entropy violin plots
  prediction_error_map.png              — D_t / ΔD̂_t / ΔD_t / signed error / skill score
"""

import argparse
import glob
import os
import re
import sys
import math

import numpy as np
import torch
from sklearn.model_selection import TimeSeriesSplit

# ── Make sure the project root is on sys.path ────────────────────────────────
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
from model_interpretability import ModelInterpreter, plot_fold_summary


# ─────────────────────────────────────────────────────────────────────────────
# Default model config — will be overridden by checkpoint or CLI args
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_CFG = dict(
    in_channels=1,
    patch_size=16,       # Will auto-detect from checkpoint
    embed_dim=64,        # Will auto-detect from checkpoint
    depth=1,             # Will auto-detect from checkpoint
    num_heads=2,         # Will auto-detect from checkpoint
    proj_drop=0.1,
    attn_drop=0.1,
    drop_path_rate=0.05,
    ls_init_value=1e-2,
    gate_init=0.0,
    use_checkpoint=False,
    attention_chunk_size=128,
)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_all_fold_checkpoints(root: str = ".") -> list[str]:
    """Return sorted list of .pth/.pt files found in root."""
    pth_files = glob.glob(os.path.join(root, "*.pth"))
    pt_files = glob.glob(os.path.join(root, "*.pt"))
    return sorted(pth_files + pt_files)


def _load_checkpoint(pth_path: str) -> dict:
    """
    Load a .pth/.pt file and return its contents as a normalised dict.

    Supports multiple formats:
      - New format (dict):  {"model_state_dict": ..., "train_mse": ..., ...}
      - With model_config:  also has "model_config" for auto-detection
      - Old format (bare state_dict):  {"pos_embed": ..., ...}
    """
    raw = torch.load(pth_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw  # new format

    # Old format: the file IS the state_dict
    return {
        "model_state_dict": raw,
        "train_loss": [],
        "val_loss": [],
        "train_r2": [],
        "val_r2_history": [],
        "scaler_mean": None,
        "scaler_std": None,
    }


def _infer_model_config_from_checkpoint(state_dict: dict) -> dict:
    """
    Infer model configuration from state_dict keys and tensor shapes.
    Returns a dict with detected values, or None for values that can't be inferred.
    """
    config = {}
    
    # Infer depth from number of blocks
    block_keys = [k for k in state_dict.keys() if k.startswith("blocks.")]
    if block_keys:
        max_block_idx = max(int(k.split(".")[1]) for k in block_keys if "blocks." in k)
        config["depth"] = max_block_idx + 1
    
    # Infer embed_dim from pos_embed shape
    if "pos_embed" in state_dict:
        config["embed_dim"] = state_dict["pos_embed"].shape[-1]
    
    # Infer num_heads from gate_logit shape — gate_logit has shape (num_heads,)
    gate_key = next((k for k in state_dict.keys() if "gate_logit" in k), None)
    if gate_key:
        config["num_heads"] = state_dict[gate_key].shape[0]
    
    # Infer patch_size from grid size (pos_embed has N patches)
    if "pos_embed" in state_dict:
        num_patches = state_dict["pos_embed"].shape[1]
        grid_size = int(math.sqrt(num_patches))
        # Reverse engineer: padded_size = grid_size * patch_size
        # We know img_size=457, so find patch_size that gives this grid
        for p in [16, 20, 24, 28, 29, 30, 32]:
            padded = math.ceil(457 / p) * p
            if padded // p == grid_size:
                config["patch_size"] = p
                break
    
    return config


def _load_fold_history(
    all_pths: list[str],
) -> tuple[list[dict], list[int]] | tuple[None, None]:
    """
    Read training history from each .pt checkpoint file.

    Returns (histories, fold_numbers) or (None, None) if no history available.
    """
    histories = []
    fold_numbers = []

    for pth_path in all_pths:
        ckpt = _load_checkpoint(pth_path)
        if not ckpt.get("val_loss"):
            continue
        
        # Extract fold number from filename
        m = re.search(r"fold[_-]?(\d+)\.(pth|pt)$", os.path.basename(pth_path))
        fold_num = int(m.group(1)) if m else None
        
        histories.append({
            "train_loss": ckpt.get("train_loss"),
            "val_loss":   ckpt.get("val_loss"),
            "train_r2":   ckpt.get("train_r2"),
            # Saved as val_r2_history to avoid collision with the scalar val_r2
            "val_r2":     ckpt.get("val_r2_history"),
        })
        fold_numbers.append(fold_num)

    if not histories:
        return None, None
    return histories, fold_numbers


def _best_pth_from_history(
    all_pths: list[str],
    histories: list[dict],
    fold_numbers: list[int],
) -> str:
    """Pick the checkpoint with best validation performance."""
    final_val_r2 = []
    for fh in histories:
        val_r2_list = fh.get("val_r2")
        if val_r2_list is not None and len(val_r2_list) > 0:
            final_val_r2.append(max(val_r2_list))
        else:
            final_val_r2.append(-999)

    best_idx = final_val_r2.index(max(final_val_r2))
    
    print("\n  Fold performance ranking (by best Val R²):")
    print(f"  {'Fold':<6} {'Best R²':<12} {'File':<40}")
    print("  " + "-" * 60)
    
    for idx in sorted(range(len(final_val_r2)), key=lambda i: final_val_r2[i], reverse=True):
        fold_str = f"Fold {fold_numbers[idx]}" if fold_numbers[idx] else "Unknown"
        r2_str = f"{final_val_r2[idx]:+.4f}" if final_val_r2[idx] > -999 else "N/A"
        marker = " ← BEST" if idx == best_idx else ""
        print(f"  {fold_str:<6} {r2_str:<12} {os.path.basename(all_pths[idx]):<40} {marker}")
    
    return all_pths[best_idx]


def _load_model(pth_path: str, sector_ids: torch.Tensor, model_cfg: dict):
    """Load model from checkpoint with given configuration."""
    try:
        from transformer_optimized import SmallDataDecoderViT
        print("  Using transformer_optimized.SmallDataDecoderViT")
    except ImportError:
        from transformer import SmallDataDecoderViT
        print("  Using transformer.SmallDataDecoderViT")
        model_cfg = {k: v for k, v in model_cfg.items()
                     if k not in ['use_checkpoint', 'attention_chunk_size']}

    model = SmallDataDecoderViT(**model_cfg, sector_ids=sector_ids)

    ckpt = _load_checkpoint(pth_path)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Strict load first — if it fails (e.g. checkpoint saved with two-layer
    # Sequential head but model now has single Linear head), report clearly
    # and attempt a partial load so at least the transformer weights are usable.
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"\n  WARNING: strict load_state_dict failed:\n    {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        print("  Continuing with partial load — head weights may be randomly initialised.")

    model.eval()
    return model


def _build_sample(
    distance_matrix: np.ndarray,
    idx: int,
    scaler_X_mean: float = None,
    scaler_X_std: float = None,
    scaler_y_mean: float = None,
    scaler_y_std: float = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (x, y) tensors for a single timestep."""
    D_t = distance_matrix[idx].astype(np.float32)
    D_next = distance_matrix[idx + 1].astype(np.float32)
    delta = D_next - D_t

    # Apply scalers
    if scaler_X_mean is not None and scaler_X_std is not None:
        x_scaled = (D_t - scaler_X_mean) / scaler_X_std
    else:
        x_scaled = D_t

    if scaler_y_mean is not None and scaler_y_std is not None:
        y_scaled = (delta - scaler_y_mean) / scaler_y_std
    else:
        y_scaled = delta

    x = torch.from_numpy(x_scaled[np.newaxis, np.newaxis, :]).float()
    y = torch.from_numpy(y_scaled[np.newaxis, np.newaxis, :]).float()
    
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate interpretability plots from a trained checkpoint."
    )
    parser.add_argument(
        "--pth",
        type=str,
        default=None,
        help="Path to a specific .pth/.pt checkpoint. If not set, auto-selects best from current dir.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Directory to search for checkpoints (default: current dir).",
    )
    parser.add_argument(
        "--dm-pkl",
        type=str,
        default=None,
        help="Explicit path to IQDw{w}.pkl. If not set, uses parameters.py path.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=-1,
        help="Index in distance matrix for attention plots (default: last sample).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Which transformer block to plot (default: 0 = first block).",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Skip data-dependent plots (only gate/gamma plots).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="interp_outputs",
        help="Output directory for plots (default: ./interp_outputs).",
    )
    # Model configuration overrides
    parser.add_argument("--embed-dim", type=int, help="Override embed_dim")
    parser.add_argument("--depth", type=int, help="Override depth")
    parser.add_argument("--num-heads", type=int, help="Override num_heads")
    parser.add_argument("--patch-size", type=int, help="Override patch_size")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    # ── Step 1: load distance matrix and build sector IDs ────────────────
    print("\n── Loading distance matrix for sector IDs …")
    try:
        distance_matrix_raw = extract_distance_matrix(pkl_path=args.dm_pkl)
    except FileNotFoundError as exc:
        if not args.no_data:
            print(f"\n  ERROR: {exc}")
            print("  Pass --no-data to skip data-dependent plots, or --dm-pkl to fix path.")
            sys.exit(1)
        else:
            print("  --no-data set: skipping distance matrix load.")
            distance_matrix_raw = None

    if distance_matrix_raw is not None:
        distance_matrix_gics, tickers_gics, sector_labels = reorder_by_gics(
            distance_matrix_raw
        )
        sector_boundaries = get_gics_sector_boundaries(sector_labels)
        del distance_matrix_raw
    else:
        # Dummy values for --no-data mode
        distance_matrix_gics = None
        tickers_gics = None
        sector_labels = None
        sector_boundaries = None

    # ── Step 2: find and select checkpoint ───────────────────────────────
    if args.pth:
        if not os.path.isfile(args.pth):
            sys.exit(f"ERROR: checkpoint '{args.pth}' not found.")
        best_pth = args.pth
        fold_history = None
        print(f"\n── Using specified checkpoint: {best_pth}")
    else:
        all_pths = _find_all_fold_checkpoints(args.root)
        if not all_pths:
            sys.exit(
                f"ERROR: no .pth/.pt files found in '{args.root}'.\n"
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
                f"\n  NOTE: no training history found in checkpoints.\n"
                f"  Falling back to last checkpoint: {best_pth}"
            )

        print(f"\n  Selected checkpoint: {best_pth}")

    # ── Step 3: determine model configuration ─────────────────────────────
    print(f"\n── Determining model configuration …")
    ckpt = _load_checkpoint(best_pth)
    
    # Priority: CLI args > checkpoint config > inferred > defaults
    model_cfg = DEFAULT_MODEL_CFG.copy()
    
    # Try loading saved config from checkpoint
    if "model_config" in ckpt:
        print("  Found model_config in checkpoint")
        model_cfg.update(ckpt["model_config"])
    else:
        # Try to infer from state_dict
        print("  Inferring config from state_dict …")
        state_dict = ckpt.get("model_state_dict", ckpt)
        inferred = _infer_model_config_from_checkpoint(state_dict)
        model_cfg.update({k: v for k, v in inferred.items() if v is not None})
    
    # Apply CLI overrides
    if args.embed_dim:
        model_cfg["embed_dim"] = args.embed_dim
    if args.depth:
        model_cfg["depth"] = args.depth
    if args.num_heads:
        model_cfg["num_heads"] = args.num_heads
    if args.patch_size:
        model_cfg["patch_size"] = args.patch_size
    
    print(f"  Final configuration:")
    print(f"    patch_size  : {model_cfg['patch_size']}")
    print(f"    embed_dim   : {model_cfg['embed_dim']}")
    print(f"    depth       : {model_cfg['depth']}")
    print(f"    num_heads   : {model_cfg['num_heads']}")
    
    # Build sector IDs with correct patch_size
    if sector_labels is not None:
        sector_ids = build_patch_sector_ids(sector_labels, patch_size=model_cfg["patch_size"])
    else:
        # Dummy for --no-data mode (won't be used)
        padded = math.ceil(457 / model_cfg["patch_size"]) * model_cfg["patch_size"]
        num_patches = (padded // model_cfg["patch_size"]) ** 2
        sector_ids = torch.zeros(num_patches, dtype=torch.long)

    # ── Step 4: load model ────────────────────────────────────────────────
    print(f"\n── Loading model …")
    model = _load_model(best_pth, sector_ids, model_cfg)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Print checkpoint metrics
    best_epoch = ckpt.get("epoch")
    best_val_r2 = ckpt.get("val_r2")
    if best_val_r2 is not None and best_epoch is not None:
        beats = "✓ beats null" if best_val_r2 > 0.0 else "✗ does NOT beat null"
        print(f"  Best epoch: {best_epoch}  |  Val R²: {best_val_r2:+.6f}  [{beats}]")

    interp = ModelInterpreter(model, save_dir=args.out_dir)
    last_block = len(list(model.blocks)) - 1

    # ── Step 5: fold summary (if available) ───────────────────────────────
    if fold_history:
        print(f"\n── Plotting fold summary ({len(fold_history)} fold(s)) …")
        plot_fold_summary(
            fold_history,
            save_path=os.path.join(args.out_dir, "fold_summary.png"),
        )

    # ── Step 6: parameter-only plots ──────────────────────────────────────
    print("\n── Gate values …")
    interp.plot_gate_values(filename="gate_values.png")

    print("\n── LayerScale gammas …")
    interp.plot_layerscale_gammas(filename="layerscale_gammas.png")

    if args.no_data:
        print("\n  --no-data set: skipping attention-map and error-map plots.")
        print(f"\nDone.  All outputs saved to '{args.out_dir}/'.")
        return

    # ── Step 7: prepare data for plots ────────────────────────────────────
    scaler_X_mean = ckpt.get("scaler_X_mean")
    scaler_X_std = ckpt.get("scaler_X_std")
    scaler_y_mean = ckpt.get("scaler_y_mean")
    scaler_y_std = ckpt.get("scaler_y_std")
    scaler_mean = ckpt.get("scaler_mean")
    scaler_std = ckpt.get("scaler_std")

    T = distance_matrix_gics.shape[0]
    sample_idx = args.sample_idx if args.sample_idx >= 0 else T - 2
    print(f"\n── Preparing data (T={T}, using index {sample_idx}) …")

    # Build scaled tensors
    D = distance_matrix_gics.astype(np.float32)
    D_t = D[:-1]
    delta = D[1:] - D[:-1]

    if scaler_X_mean is not None and scaler_X_std is not None:
        X_scaled = (D_t - scaler_X_mean) / scaler_X_std
    elif scaler_mean is not None and scaler_std is not None:
        X_scaled = (D_t - scaler_mean) / scaler_std
    else:
        X_scaled = D_t

    if scaler_y_mean is not None and scaler_y_std is not None:
        y_scaled = (delta - scaler_y_mean) / scaler_y_std
    elif scaler_mean is not None and scaler_std is not None:
        y_scaled = (delta - scaler_mean) / scaler_std
    else:
        y_scaled = delta

    X_t = torch.from_numpy(X_scaled[:, np.newaxis, :]).float()
    y_t = torch.from_numpy(y_scaled[:, np.newaxis, :]).float()

    # Get validation split
    tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)
    *_, (_, last_val_idx) = tscv.split(X_t)
    X_val = X_t[last_val_idx]
    y_val = y_t[last_val_idx]

    # Null model baseline
    if scaler_y_mean is not None and scaler_y_std is not None and float(scaler_y_std) != 0.0:
        null_value = float(-scaler_y_mean) / float(scaler_y_std)
    else:
        null_value = 0.0

    sample_x, sample_y = _build_sample(
        distance_matrix_gics, sample_idx,
        scaler_X_mean, scaler_X_std,
        scaler_y_mean, scaler_y_std,
    )
    baseline_pred = torch.full_like(sample_y, null_value)
    baseline_val_full = torch.full_like(y_val, null_value)

    # ── Step 8: data-dependent plots ──────────────────────────────────────
    print(f"\n── Attention maps (block {args.layer}) …")
    interp.plot_attention_maps(
        sample_x, layer=args.layer,
        filename=f"attention_maps_block{args.layer}.png",
    )

    # Only plot last block separately if model has multiple layers
    if last_block > 0:
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

    if last_block > 0:
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

    print(f"\n{'='*70}")
    print(f"Done.  All outputs saved to '{args.out_dir}/'.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
