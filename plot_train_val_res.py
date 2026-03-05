import matplotlib.pyplot as plt
import numpy as np

def plot_train_val_res(all_fold_history):

    fig, axes = plt.subplots(2, 9, figsize=(15, 6))
    axes = axes.flatten()
    for i, axs in enumerate(axes):
        if i <= 8: #Plot MSE
            y_data = all_fold_history[i]['train_mse']
            x_data = np.arange(1, len(y_data) + 1)
            axs.plot(x_data, y_data, label=f'Fold {i+1} Train', alpha=0.3, color='blue')

            y_data = all_fold_history[i]['val_mse']
            x_data = np.arange(1, len(y_data) + 1)
            axs.plot(x_data, y_data, label=f'Fold {i+1} Val', linestyle='--', alpha=0.8)

            if i == 0:
                axs.set_ylabel('Mean Squared Error (MSE)')

        elif i >= 9 and i <= 17: #Plot R^2

            y_data = [t.item() for t in all_fold_history[i-9]['train_r2']]
            x_data = np.arange(1, len(y_data) + 1)
            axs.plot(x_data, y_data, alpha=0.3, color='green')

            y_data = [t.item() for t in all_fold_history[i-9]['val_r2']]
            x_data = np.arange(1, len(y_data) + 1)
            axs.plot(x_data, y_data, linestyle='--', alpha=0.8)

            axs.set_xlabel(f'Epoch {i-8}')

            if i == 9:
                axs.set_ylabel('$R^2$ Score')
            # elif i == 13:
            #     axs.set_xlabel('Epoch')

        else:
            print("plot indexing error.")

    plt.tight_layout()
    plt.rcParams["axes.unicode_minus"] = True
    plt.rcParams["font.family"] = "sans-serif"
    plt.savefig('Vision_Forecaster_Train_Val_Result.pdf')  # bbox_inches='tight'
    plt.savefig('Vision_Forecaster_Train_Val_Result.png')  # , bbox_inches='tight'