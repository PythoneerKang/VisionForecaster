# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#from torch.amp import GradScaler
from parameters import *
from extract_distance_matrices import *
from transformer import *
from training_and_validation_functions import *
from plot_train_val_res import *
#from scratch import *


if __name__ == "__main__":
    # First, extract distance matrices from pkl file.
    # Dr. Cheong says only focus on w = 35 days.
    distance_matrix = extract_distance_matrix()

    print(
        "Check for NAN values in distance matrices, (False -> no NAN, True otherwise.): ",
        np.isnan(distance_matrix).any(),
    )
    print(
        "Distance matrix shape: (num_of_trading_days, num_of_stocks, num_of_stocks): ",
        distance_matrix.shape,
    )

    # Training and validation with multi-fold CV (builds each fold on-the-fly)
    model_path, all_fold_history = diff_model_multi_fold_cv_train_test(distance_matrix)

    #model_path = "best_model_w180_fold_9.pth"

    # ── Add to main.py after diff_model_multi_fold_cv_train_test() ──────────────

    from model_interpretability import ModelInterpreter, plot_fold_summary
    from transformer import SmallDataDecoderViT

    # 1. Training summary across all folds
    plot_fold_summary(all_fold_history, save_path="fold_summary.png")

    # 2. Load the best model
    best_model = SmallDataDecoderViT(
        in_channels=1, embed_dim=192, depth=6, num_heads=3,
        proj_drop=0.1, drop_path_rate=0.05,
    )
    best_model.load_state_dict(torch.load(model_path, map_location="cpu"))

    interp = ModelInterpreter(best_model, save_dir=".")

    # 3. Get one sample from the last val_loader fold (rebuild quickly)
    import torch
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import TimeSeriesSplit

    X = distance_matrix[:-1][:, np.newaxis, :]
    y = distance_matrix[1:][:, np.newaxis, :]
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)
    *_, (_, last_val_idx) = tscv.split(X_t)
    sample_x = X_t[last_val_idx[:1]]   # (1, 1, 457, 457)
    sample_y = y_t[last_val_idx[:1]]

    # 4. Generate all interpretation plots
    interp.plot_attention_maps(sample_x, layer=0)
    interp.plot_attention_maps(sample_x, layer=best_model.blocks.__len__() - 1,
                            filename="attention_maps_last_block.png")
    interp.plot_attention_maps_overlay(sample_x, layer=0)
    interp.plot_attention_maps_overlay(sample_x, layer=best_model.blocks.__len__() - 1,
                                   filename="attention_maps_overlay_last_block.png")
    interp.plot_mean_attention_distance(sample_x)
    interp.plot_layerscale_gammas()
    interp.plot_attention_temperatures()
    interp.plot_locality_weights()
    interp.plot_locality_bias_scale(sample_x)   # ← diagnoses overfocusing risk
    interp.plot_prediction_error_map(sample_x, sample_y)

    # plot_train_val_res(all_fold_history)

    # #Finally, testing.
    # evaluate_model(model,test_loader,criterion)