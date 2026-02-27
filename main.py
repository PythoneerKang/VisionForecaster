# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#from torch.amp import GradScaler
from parameters import *
from extract_distance_matrices import *
from preprocessing import *
from transformer import *
from training_and_validation_functions import *
from plot_train_val_res import *
#from scratch import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #First, extract distance matrices from pkl file.
    #Dr. Cheong says only focus on w = 35 days.
    distance_matrix = extract_distance_matrix()

    print("Check for NAN values in distance matrices, (False -> no NAN, True otherwise.): ", np.isnan(distance_matrix).any())
    print("Distance matrix shape: (num_of_trading_days, num_of_stocks, num_of_stocks): ", distance_matrix.shape)

    if p.save_loaders:
        #Next, pre-process data into training, validation, and testing sets.
        train_loaders, test_loaders = generate_multi_fold_cv_dataloaders(distance_matrix) #, val_loader,
        loaders = {
        'train': train_loaders,
        'test': test_loaders}
        torch.save(loaders, '../LFS/{}/dataloaders.pt'.format(repo_name))

    import torch.serialization
    from torch.utils.data.dataloader import DataLoader

    # Register the specific class as safe
    torch.serialization.add_safe_globals([DataLoader])
    
    loaded_loaders = torch.load('../LFS/{}/dataloaders.pt'.format(repo_name), weights_only=False)
    train_loaders = loaded_loaders['train']
    test_loaders = loaded_loaders['test']

    printshape(train_loaders,test_loaders)

    #train_loss = train_step(next(iter(train_loaders[0])),next(iter(train_loaders[0])))

    #After that, begin the transformer training, validation, and testing process.
    #Training and validation first.
    model_path, all_fold_history = diff_model_multi_fold_cv_train_test(train_loaders, test_loaders) #

    plot_train_val_res(all_fold_history)

    # #Finally, testing.
    # evaluate_model(model,test_loader,criterion)