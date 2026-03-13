from parameters import *
from extract_distance_matrices import (
    extract_distance_matrix,
    reorder_by_gics,
    get_gics_sector_boundaries,
    build_patch_sector_ids,
)
from transformer import SmallDataDecoderViT
from training_and_validation_functions import diff_model_multi_fold_cv_train_test

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Model hyperparameters — single source of truth shared by training and
# interpretability so the loaded checkpoint always matches training config.
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


if __name__ == "__main__":
    # ── Step 1: Extract distance matrices from pkl file ───────────────────────
    distance_matrix = extract_distance_matrix()

    print(
        "Check for NAN values in distance matrices, (False -> no NAN, True otherwise.): ",
        np.isnan(distance_matrix).any(),
    )
    print(
        "Distance matrix shape (num_of_trading_days, num_of_stocks, num_of_stocks): ",
        distance_matrix.shape,
    )

    # ── Step 2: Reorder stocks by GICS sector ────────────────────────────────
    distance_matrix_gics, tickers_gics, sector_labels = reorder_by_gics(
        distance_matrix
    )

    sector_boundaries = get_gics_sector_boundaries(sector_labels)

    print("\nGICS reordering applied. Sector block sizes:")
    for name, start, end in sector_boundaries:
        print(f"  {name:<30}  stocks [{start:>3}, {end:>3})  n={end - start}")
    print()

    del distance_matrix

    # ── Step 3: Build patch-level sector IDs for SectorGPSA ──────────────────
    # Each of the 841 patches (29×29 grid) is assigned the GICS sector index
    # of its dominant stock row.  This is the positional prior used by
    # SectorGPSA: each patch attends uniformly over all patches in the same
    # sector at the start of training (gate ≈ 0.88 positional).
    sector_ids = build_patch_sector_ids(sector_labels)

    print(f"Patch sector IDs built: shape={tuple(sector_ids.shape)}, "
          f"unique sectors={sector_ids.unique().numel()}")
    print()

    # ── Step 4: Train with multi-fold CV ─────────────────────────────────────
    model_path, all_fold_history = diff_model_multi_fold_cv_train_test(
        distance_matrix_gics, sector_ids
    )

    # ── Step 5: Interpretability ──────────────────────────────────────────────
    from model_interpretability import ModelInterpreter, plot_fold_summary

    # 5a. Training summary across all folds
    plot_fold_summary(all_fold_history, save_path="fold_summary.png")

    # 5b. Load the best model — use the same config as training to ensure
    #     the architecture and gate_init match the saved checkpoint exactly.
    best_model = SmallDataDecoderViT(
        **MODEL_CFG,
        sector_ids=sector_ids,
    )
    best_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    best_model.load_state_dict(best_ckpt["model_state_dict"] if isinstance(best_ckpt, dict) and "model_state_dict" in best_ckpt else best_ckpt)

    interp = ModelInterpreter(best_model, save_dir=".")

    # 5c. Rebuild one sample from the last validation fold
    # IMPORTANT: apply the fold-wise scaler from the chosen checkpoint so
    # interpretability metrics match what the model was trained/evaluated on.
    scaler_mean = best_ckpt.get("scaler_mean") if isinstance(best_ckpt, dict) else None
    scaler_std  = best_ckpt.get("scaler_std")  if isinstance(best_ckpt, dict) else None
    if scaler_mean is not None and scaler_std is not None:
        dm_for_eval = (distance_matrix_gics - scaler_mean) / scaler_std
    else:
        dm_for_eval = distance_matrix_gics

    X   = dm_for_eval[:-1][:, np.newaxis, :]
    y   = dm_for_eval[1:][:, np.newaxis, :]
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)
    *_, (_, last_val_idx) = tscv.split(X_t)

    # Full validation fold — used for val-set MSE comparison in the error map,
    # matching the val_mse numbers logged during training exactly.
    X_val   = X_t[last_val_idx]          # (N_val, 1, 457, 457)
    y_val   = y_t[last_val_idx]

    # Single display sample: last day of the validation fold (never seen in training)
    sample_x = X_val[-1:]                # (1, 1, 457, 457)
    sample_y = y_val[-1:]

    # 5d. Generate all interpretation plots
    interp.plot_attention_maps(sample_x, layer=0)
    interp.plot_attention_maps(
        sample_x,
        layer=len(best_model.blocks) - 1,
        filename="attention_maps_last_block.png",
    )
    interp.plot_attention_maps_overlay(sample_x, layer=0)
    interp.plot_attention_maps_overlay(
        sample_x,
        layer=len(best_model.blocks) - 1,
        filename="attention_maps_overlay_last_block.png",
    )
    interp.plot_gate_values()
    interp.plot_mean_attention_distance(sample_x)
    interp.plot_layerscale_gammas()
    interp.plot_attention_weights(sample_x)
    # X_val / y_val passed so MSE is computed over the full validation fold,
    # giving an apples-to-apples comparison with the baseline and train logs.
    interp.plot_prediction_error_map(
        sample_x,
        sample_y,
        tickers=tickers_gics,
        sector_boundaries=sector_boundaries,
        X_val=X_val,
        y_val=y_val,
    )
