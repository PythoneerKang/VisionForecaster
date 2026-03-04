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

    plot_train_val_res(all_fold_history)

    # #Finally, testing.
    # evaluate_model(model,test_loader,criterion)