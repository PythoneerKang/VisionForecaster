def plot_train_val_res(all_fold_history):

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i, axs in enumerate(axes):
        if i <= 4: #Plot MSE
            axs.plot(all_fold_history[i]['train_mse'], label=f'Fold {i+1} Train', alpha=0.3, color='blue')
            axs.plot(all_fold_history[i]['val_mse'], label=f'Fold {i+1} Val', linestyle='--', alpha=0.8)

            if i == 0:
                axs.set_ylabel('Mean Squared Error (MSE)')

        elif i >= 5 and i <= 9: #Plot R^2
            axs.plot(all_fold_history[i]['train_r2'], alpha=0.3, color='green')
            axs.plot(all_fold_history[i]['val_r2'], linestyle='--', alpha=0.8)

            if i == 5:
                axs.set_ylabel('$R^2$ Score')
            elif i == 7:
                axs.set_xlabel('Epoch')

        else:
            print("plot indexing error.")

    plt.tight_layout()
    plt.rcParams["axes.unicode_minus"] = True
    plt.rcParams["font.family"] = "sans-serif"
    plt.savefig('Vision_Forecaster_Train_Val_Result.pdf')  # bbox_inches='tight'
    plt.savefig('Vision_Forecaster_Train_Val_Result.png')  # , bbox_inches='tight'