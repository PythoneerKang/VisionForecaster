import matplotlib.pyplot as plt
import numpy as np


def plot_train_val_res(all_fold_history):
    """
    Plot training and validation MSE and R² curves for all 9 folds.

    NOTE: This function is superseded by model_interpretability.plot_fold_summary()
    which produces a richer summary. This simpler version is kept as a backup.

    Layout: 2 rows × 9 columns
      Row 1 (axes 0–8)  : MSE curves per fold
      Row 2 (axes 9–17) : R² curves per fold
    """
    # Set rcParams BEFORE creating figures so they apply to the saved output
    plt.rcParams["axes.unicode_minus"] = True
    plt.rcParams["font.family"] = "sans-serif"

    fig, axes = plt.subplots(2, 9, figsize=(15, 6))
    axes = axes.flatten()

    for i, axs in enumerate(axes):
        if i <= 8:  # Row 1: MSE curves
            fold_idx = i
            y_data = all_fold_history[fold_idx]['train_mse']
            x_data = np.arange(1, len(y_data) + 1)
            axs.plot(x_data, y_data, label=f'Fold {fold_idx+1} Train',
                     alpha=0.3, color='blue')

            y_data = all_fold_history[fold_idx]['val_mse']
            x_data = np.arange(1, len(y_data) + 1)
            axs.plot(x_data, y_data, label=f'Fold {fold_idx+1} Val',
                     linestyle='--', alpha=0.8)

            axs.set_title(f'Fold {fold_idx+1}', fontsize=8)
            axs.set_xlabel('Epoch', fontsize=7)
            if i == 0:
                axs.set_ylabel('Mean Squared Error (MSE)')
            axs.legend(fontsize=6)

        elif 9 <= i <= 17:  # Row 2: R² curves
            fold_idx = i - 9
            y_data = [t.item() if hasattr(t, 'item') else t
                      for t in all_fold_history[fold_idx]['train_r2']]
            x_data = np.arange(1, len(y_data) + 1)
            axs.plot(x_data, y_data, alpha=0.3, color='green',
                     label=f'Fold {fold_idx+1} Train')

            y_data = [t.item() if hasattr(t, 'item') else t
                      for t in all_fold_history[fold_idx]['val_r2']]
            x_data = np.arange(1, len(y_data) + 1)
            axs.plot(x_data, y_data, linestyle='--', alpha=0.8,
                     label=f'Fold {fold_idx+1} Val')

            axs.set_title(f'Fold {fold_idx+1}', fontsize=8)
            axs.set_xlabel('Epoch', fontsize=7)   # x-axis is epochs, not fold index
            if i == 9:
                axs.set_ylabel('$R^2$ Score')
            axs.axhline(0, color='gray', linewidth=0.7, linestyle=':')

        else:
            print(f"plot indexing error at i={i}.")

    plt.tight_layout()
    plt.savefig('Vision_Forecaster_Train_Val_Result.pdf')
    plt.savefig('Vision_Forecaster_Train_Val_Result.png')
    plt.close(fig)  # Release memory — important for long HPC runs
