import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import torch
from torch.utils.data import TensorDataset, DataLoader
import parameters as p

def get_shape(dataloader):
    num_samples = len(dataloader.dataset)
    # Iterate once to get a single batch
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    single_image_shape = images.shape[1:]
    single_label_shape = labels.shape[1:] if labels.dim() > 1 else labels.shape[0:]

    return (num_samples, *single_image_shape), (num_samples, *single_label_shape)

def generate_multi_fold_cv_dataloaders(distance_matrix):

    trainloaders = []
    testloaders = []

    X = distance_matrix[:-1][:, np.newaxis, :]
    y = distance_matrix[1:][:, np.newaxis, :]
    print(f"Non-padded data shape: {X.shape}, {y.shape}")

    tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)  # n_splits determines the number of splits

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold+1}:")
        #print(f"  Train indices: {train_index}")
        #print(f"  Test indices: {test_index}")

        # Access the data for the current fold
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]

        # Use torch.from_numpy() rather than torch.tensor() to avoid an
        # unnecessary data copy and the associated UserWarning in PyTorch ≥ 1.8.
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_cv).float(),
            torch.from_numpy(y_train_cv).float(),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test_cv).float(),
            torch.from_numpy(y_test_cv).float(),
        )

        train_loader = DataLoader(train_dataset, batch_size=p.BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=p.BATCH_SIZE)

        Training_set_shape = get_shape(train_loader)
        Testing_set_shape = get_shape(test_loader)

        print(f"Training set shape: {Training_set_shape}")
        print(f"Testing set shape: {Testing_set_shape}")

        trainloaders.append(train_loader)
        testloaders.append(test_loader)

    return trainloaders, testloaders

def printshape(train_loader,test_loader):
    for fold in range(9):
        print(f"Fold {fold+1}:")

        for inputs, _ in train_loader[fold]:
            if torch.isnan(inputs).sum() > 0:
                raise ValueError(f"NaN detected in train_loader fold {fold} data!")

        for inputs, _ in test_loader[fold]:
            if torch.isnan(inputs).sum() > 0:
                raise ValueError(f"NaN detected in test_loader fold {fold} data!")

        Training_set_shape = get_shape(train_loader[fold])
        Testing_set_shape = get_shape(test_loader[fold])

        print(f"Training set shape: {Training_set_shape}")
        print(f"Testing set shape: {Testing_set_shape}")
