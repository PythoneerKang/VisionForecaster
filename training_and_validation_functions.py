import parameters as p
import torch
from torchmetrics import R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformer import *

#from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, path='best_model.pt'):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path) # Save best weights
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True # Signal to stop
            return False


def r_squared_score(predictions, targets):
    """
    Calculates the R-squared (coefficient of determination) score in PyTorch.

    Args:
        predictions (torch.Tensor): The predicted values from the model.
        targets (torch.Tensor): The actual ground truth values.

    Returns:
        torch.Tensor: The R-squared score.
    """
    # Calculate the mean of the target values
    targets_mean = torch.mean(targets)

    # Calculate the Total Sum of Squares (SStot)
    ss_tot = torch.sum((targets - targets_mean) ** 2)

    # Calculate the Sum of Squared Residuals (SSres)
    ss_res = torch.sum((targets - predictions) ** 2)

    # Calculate R-squared
    r2 = 1 - ss_res / ss_tot

    return r2

def train_with_validation(model, train_loader, val_loader,fold, epochs=100):

    device =  torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=1) #Change to 10

    fold_history = {'train_mse': [], 'val_mse': [], 'train_r2': [], 'val_r2': []}

    for epoch in range(1,epochs+1):
        print(f"----- Epoch {epoch} -----")
        # --- TRAINING PHASE ---
        model.train()
        train_targets = []
        train_predictions = []
        train_loss = 0
        print("Training begins")
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            #print(f"Batch {batch_idx + 1}: Features shape {inputs.shape}, Labels shape {labels.shape}")
            x, y = inputs.to(device), labels.to(device)
        # for x, y in train_loader:
        #     x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            #labels = y
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_targets.append(y.cpu())
            train_predictions.append(outputs.cpu())
            train_loss += loss.item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_targets = []
        val_predictions = []
        val_loss = 0
        print("Validation begins")
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(train_loader):
                #print(f"Batch {batch_idx + 1}: Features shape {x.shape}, Labels shape {y.shape}")
                x, y = x.to(device), y.to(device)
                #labels = y
                outputs = model(x)
                val_targets.append(y.cpu())
                val_predictions.append(outputs.cpu())
                val_loss += criterion(outputs, y).item()


        # --- PRINT RESULTS ---

        train_targets = torch.cat(train_targets)
        train_predictions = torch.cat(train_predictions)
        r2score = R2Score(multioutput='uniform_average')
        r2_value = r2score(train_predictions.view(-1), train_targets.view(-1))
        epoch_training_r2_score = r2_value #r_squared_score(all_predictions, all_targets)

        val_targets = torch.cat(val_targets)
        val_predictions = torch.cat(val_predictions)
        r2score = R2Score(multioutput='uniform_average')
        r2_value = r2score(val_predictions.view(-1), val_targets.view(-1))
        epoch_validation_r2_score = r2_value

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"----- Train/Validation results -----")
        print(f"Epoch {epoch}: Train Loss {avg_train:.6f} | Val Loss {avg_val:.6f}")
        print(f"Epoch {epoch}: Train R^2: {epoch_training_r2_score * 100:.4f}% | Val R^2: {epoch_validation_r2_score * 100:.4f}%")
        print("-"*20)

        # Store Metrics for plotting
        fold_history['train_mse'].append(avg_train)
        fold_history['val_mse'].append(avg_val)
        fold_history['train_r2'].append(epoch_training_r2_score)
        fold_history['val_r2'].append(epoch_validation_r2_score)

        if stopper(avg_val, model):
            print("Early stopping triggered. Loading best model...")
            model.load_state_dict(torch.load('best_model.pt'))
            break
    
    model_path = f'model_fold_{fold}.pth'
    torch.save(model.state_dict(), model_path) # [1]
        
    return model_path, avg_train, avg_val, epoch_training_r2_score, epoch_validation_r2_score, fold_history

def diff_model_multi_fold_cv_train_test(trainloaders, testloaders): #

    fold = 1
    fold_models = []
    fold_result = np.zeros((5,4))

    all_fold_history = [] #For final training/val result plots.

    for trainloader,val_loader in zip(trainloaders,testloaders):
        print(10*"=", f"Fold={fold}", 10*"=")

        #Instantiate the transformer model for each fold
        model = SmallDataDecoderViT(
        in_channels=1,
        embed_dim=64, depth=8, num_heads=8,
        proj_drop=0.1, drop_path_rate=0.1)

        #This is the original Decoder-only ViT without SPT and LSA
        # model = DecoderOnlyViT(
        # in_channels=1,
        # img_size=457,
        # patch_size=16,          # try 16 or 32
        # embed_dim=64,          # small for smoke test
        # depth=4,
        # num_heads=8,
        # mlp_ratio=4.0)
        path, train_mse, val_mse, train_r2, val_r2, fold_history = train_with_validation(model, trainloader, val_loader,fold, epochs=p.num_epochs)
        fold_models.append(path)
        fold_result[fold-1] = np.array([
                                train_mse.detach().cpu().numpy(), 
                                val_mse.detach().cpu().numpy(), 
                                train_r2.detach().cpu().numpy(), 
                                val_r2.detach().cpu().numpy()
                                ])
        fold += 1

        all_fold_history.append(fold_history)
   
    #Compare among the folds, choose the one with the highest val R^2 and/or lowest val MSE
    model_lowest_val_mse = np.argmax(fold_result[:,1])
    model_highest_val_r2 = np.argmax(fold_result[:,3])
    if model_lowest_val_mse == model_highest_val_r2:
        print("Lowest val_mse model = highest val_r2 model, all good.")
        print(f"Model {model_lowest_val_mse} has the lowest val_MSE and the highest val_R^2.")
    else:
        print("Lowest val_mse model != highest val_r2 model, choose lowest mse model.")
        print(f"Model {model_lowest_val_mse} has the lowest val_MSE.")
        print(f"Model {model_highest_val_r2} has the highest val_R^2.")
        return 

    return fold_models[model_lowest_val_mse], all_fold_history 
    
# To load model, define model class first.
#model = VisionForecaster(img_size=p.IMG_SIZE , patch_size=p.PATCH_SIZE, in_chans=p.CHANNELS, embed_dim=p.EMBED_DIM, depth=p.DEPTH, heads=p.HEADS, mlp_dim=p.MLP_DIM)
#PATH = 'your_model_path.pth' i.e., output of diff_model_multi_fold_cv_train_test() above.
#model.load_state_dict(torch.load(PATH))

