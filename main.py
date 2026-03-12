from parameters import *
from extract_distance_matrices import (
    extract_distance_matrix,
    reorder_by_gics,
    get_gics_sector_boundaries,
)
from transformer import *
from training_and_validation_functions import *


if __name__ == "__main__":
    # ── Step 1: Extract distance matrices from pkl file ───────────────────────
    # Dr. Cheong says only focus on w = 35 days.
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
    # Grouping stocks by GICS sector makes spatial proximity in the 457×457
    # matrix financially meaningful: nearby entries belong to the same industry,
    # so the LSA locality bias captures genuine sector co-movement structure
    # rather than arbitrary index adjacency.
    #
    # reorder_by_gics() returns:
    #   distance_matrix_gics : (T, 457, 457) with rows/cols permuted by sector
    #   tickers_gics         : list[str] of tickers in the new GICS order
    #   sector_labels        : list[str] of sector name per stock (same order)
    distance_matrix_gics, tickers_gics, sector_labels = reorder_by_gics(
        distance_matrix
    )

    # Sector block boundaries — used later for annotating interpretability plots
    sector_boundaries = get_gics_sector_boundaries(sector_labels)

    print("\nGICS reordering applied. Sector block sizes:")
    for name, start, end in sector_boundaries:
        print(f"  {name:<30}  stocks [{start:>3}, {end:>3})  n={end - start}")
    print()

    # The original (non-GICS) matrix is no longer needed
    del distance_matrix

    # ── Step 3: Train with multi-fold CV ─────────────────────────────────────
    # All downstream code receives the GICS-reordered matrix transparently —
    # the model architecture and training loop are unchanged.
    model_path, all_fold_history = diff_model_multi_fold_cv_train_test(
        distance_matrix_gics
    )

    # ── Step 4: Interpretability ──────────────────────────────────────────────

    from model_interpretability import ModelInterpreter, plot_fold_summary

    # 4a. Training summary across all folds
    plot_fold_summary(all_fold_history, save_path="fold_summary.png")

    # 4b. Load the best model.
    #     ls_init_value is passed explicitly for consistency with training, even
    #     though it only affects __init__ and gamma values come from the loaded weights.
    best_model = SmallDataDecoderViT(
        in_channels=1,
        embed_dim=192,
        depth=6,
        num_heads=3,
        proj_drop=0.1,
        drop_path_rate=0.05,
        ls_init_value=1e-2,
    )
    best_model.load_state_dict(torch.load(model_path, map_location="cpu"))

    interp = ModelInterpreter(best_model, save_dir=".")

    # 4c. Rebuild one sample from the last validation fold.
    #     TimeSeriesSplit config must match diff_model_multi_fold_cv_train_test exactly.
    X = distance_matrix_gics[:-1][:, np.newaxis, :]
    y = distance_matrix_gics[1:][:, np.newaxis, :]
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)
    *_, (_, last_val_idx) = tscv.split(X_t)
    sample_x = X_t[last_val_idx[:1]]   # (1, 1, 457, 457)
    sample_y = y_t[last_val_idx[:1]]

    # 4d. Generate all interpretation plots.
    #     prediction_error_map receives GICS metadata so it can annotate sector
    #     boundaries on the heatmap axes.
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
    interp.plot_mean_attention_distance(sample_x)
    interp.plot_layerscale_gammas()
    interp.plot_attention_temperatures()
    interp.plot_locality_weights()
    interp.plot_locality_bias_scale(sample_x)
    interp.plot_prediction_error_map(
        sample_x,
        sample_y,
        tickers=tickers_gics,
        sector_boundaries=sector_boundaries,
    )
