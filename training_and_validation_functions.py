import parameters as p
import torch
from torchmetrics import R2Score
from transformer import *
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset
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


# NOTE: r_squared_score() below is legacy code kept for reference.
# Active code uses torchmetrics.R2Score instead (handles edge cases better).
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


def _build_optimizer(model):
    """
    Build an AdamW optimizer with three parameter groups:

      1. decay_params   — weight matrices (lr=1e-4, weight_decay=1e-2)
      2. nodecay_params — biases + LayerNorm weights (lr=1e-4, weight_decay=0)
      3. gamma_params   — LayerScale gammas (lr=1e-3, weight_decay=0)
                          10× higher LR so gammas escape their near-zero init.

    Without a boosted LR for gammas they receive gradients that are
    O(gamma) ≈ O(1e-2) in magnitude, which at lr=1e-4 produces parameter
    updates of ~1e-6 — far too small to move the gammas meaningfully.
    """
    decay_params, nodecay_params, gamma_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gamma" in name:                  # LayerScale gammas
            gamma_params.append(param)
        elif param.ndim < 2:                 # biases, LayerNorm weight/bias
            nodecay_params.append(param)
        else:                                # weight matrices
            decay_params.append(param)

    # Sanity-check: every parameter must land in exactly one group
    n_total   = sum(param_.numel() for param_ in model.parameters() if param_.requires_grad)
    n_grouped = (sum(param_.numel() for param_ in decay_params) +
                 sum(param_.numel() for param_ in nodecay_params) +
                 sum(param_.numel() for param_ in gamma_params))
    assert n_total == n_grouped, (
        f"Parameter group mismatch: {n_total} total vs {n_grouped} grouped."
    )

    print(f"  Optimizer param groups:")
    print(f"    decay    : {sum(param_.numel() for param_ in decay_params):>10,} params  lr=1e-4  wd=1e-2")
    print(f"    no-decay : {sum(param_.numel() for param_ in nodecay_params):>10,} params  lr=1e-4  wd=0")
    print(f"    gamma    : {sum(param_.numel() for param_ in gamma_params):>10,} params  lr=1e-3  wd=0  ← 10× boost")

    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "lr": 1e-4, "weight_decay": 1e-2},
        {"params": nodecay_params, "lr": 1e-4, "weight_decay": 0.0},
        {"params": gamma_params,   "lr": 1e-3, "weight_decay": 0.0},
    ], lr=1e-4)

    return optimizer


def _check_gamma_gradients(model):
    """
    Print gradient norms for all LayerScale gamma parameters.
    Call once after the first backward pass to verify gammas are receiving
    meaningful gradients. A norm of 0.000000 indicates a disconnected path.
    """
    print("\n  ── LayerScale gamma gradient check (epoch 1, batch 1) ──")
    any_printed = False
    for name, param in model.named_parameters():
        if "gamma" in name:
            if param.grad is not None:
                print(f"    {name:50s}  grad norm = {param.grad.norm().item():.6f}")
            else:
                print(f"    {name:50s}  grad = None  ← not connected!")
            any_printed = True
    if not any_printed:
        print("    WARNING: no gamma parameters found in model!")
    print("  ── end gamma check ──\n")


def train_with_validation(model, train_loader, val_loader, fold, epochs=100):

    # Device selection: for Intel CPU-only HPC runs, p.USE_GPU is False so
    # we always stay on CPU even if a CUDA device is visible.
    device = torch.device("cuda" if p.USE_GPU and torch.cuda.is_available() else "cpu")
    model.to(device)

    # ── Optimizer: separate LR for LayerScale gammas ──────────────────────
    optimizer = _build_optimizer(model)

    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=10)

    fold_history = {'train_mse': [], 'val_mse': [], 'train_r2': [], 'val_r2': []}

    # Track whether we've done the gamma gradient check yet
    _gamma_check_done = False

    for epoch in range(1, epochs + 1):
        print(f"----- Epoch {epoch} -----")
        # --- TRAINING PHASE ---
        model.train()
        # Store detached CPU copies only (no computation graph)
        train_targets = []
        train_predictions = []
        train_loss = 0
        print("Training begins")
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            x, y = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            # ── Gamma gradient check: runs once on the very first batch ──
            if not _gamma_check_done:
                _check_gamma_gradients(model)
                _gamma_check_done = True

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Move to CPU and detach so we don't keep the computation graph
            train_targets.append(y.detach().cpu())
            train_predictions.append(outputs.detach().cpu())
            train_loss += loss.item()

        # --- VALIDATION PHASE ---
        model.eval()
        # Store detached CPU copies only (no computation graph)
        val_targets = []
        val_predictions = []
        val_loss = 0
        print("Validation begins")
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                outputs = model(x)

                # Move to CPU and detach (no graph under no_grad, but keep it consistent)
                val_targets.append(y.detach().cpu())
                val_predictions.append(outputs.detach().cpu())
                val_loss += criterion(outputs, y).item()

        # --- PRINT RESULTS ---

        train_targets = torch.cat(train_targets)
        train_predictions = torch.cat(train_predictions)
        r2score = R2Score(multioutput='uniform_average')
        r2_value = r2score(train_predictions.view(-1), train_targets.view(-1))
        epoch_training_r2_score = r2_value

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

        # ── Log mean LayerScale gamma values every 10 epochs ─────────────
        if epoch % 10 == 0 or epoch == 1:
            gamma_vals = {
                name: param.detach().cpu().mean().item()
                for name, param in model.named_parameters()
                if "gamma" in name
            }
            if gamma_vals:
                mean_gamma = sum(gamma_vals.values()) / len(gamma_vals)
                min_gamma  = min(gamma_vals.values())
                max_gamma  = max(gamma_vals.values())
                print(f"  LayerScale γ — mean: {mean_gamma:.6f}  "
                      f"min: {min_gamma:.6f}  max: {max_gamma:.6f}")

        print("-" * 20)

        # Store Metrics for plotting
        fold_history['train_mse'].append(avg_train)
        fold_history['val_mse'].append(avg_val)
        fold_history['train_r2'].append(epoch_training_r2_score)
        fold_history['val_r2'].append(epoch_validation_r2_score)

        if stopper(avg_val, model):
            print("Early stopping triggered. Loading best model...")
            model.load_state_dict(torch.load(stopper.path))
            break

    # Always load best checkpoint before saving the fold model —
    # if early stopping didn't trigger, the loop ends on the final epoch which
    # may not be the best val epoch. Loading stopper.path guarantees best weights.
    model.load_state_dict(torch.load(stopper.path))

    # Save best weights (from EarlyStopping checkpoint) as the fold model.
    # This ensures model_fold_{fold}.pth always contains the best val weights,
    # regardless of whether early stopping triggered or training ran to completion.
    model_path = f"model_fold_{fold}.pth"
    torch.save(model.state_dict(), model_path)

    return model_path, fold_history


def diff_model_multi_fold_cv_train_test(distance_matrix: np.ndarray):
    """
    Perform multi-fold CV training using a single shared base tensor and
    constructing DataLoaders one fold at a time to keep memory usage low.
    """

    # Configure PyTorch threading for CPU training on HPC.
    if p.TORCH_NUM_THREADS is not None:
        torch.set_num_threads(p.TORCH_NUM_THREADS)
    if p.TORCH_NUM_INTEROP_THREADS is not None:
        torch.set_num_interop_threads(p.TORCH_NUM_INTEROP_THREADS)

    # Build base tensors once (float32, on CPU)
    X = distance_matrix[:-1][:, np.newaxis, :]
    y = distance_matrix[1:][:, np.newaxis, :]

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)

    fold = 1
    fold_models = []
    all_fold_history = []  # For final training/val result plots.

    for train_index, val_index in tscv.split(X_tensor):
        print(10 * "=", f"Fold={fold}", 10 * "=")

        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=p.BATCH_SIZE,
            shuffle=False,
            num_workers=p.NUM_WORKERS,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=p.BATCH_SIZE,
            shuffle=False,
            num_workers=p.NUM_WORKERS,
            pin_memory=False,
        )

        # Instantiate the transformer model for each fold.
        # ls_init_value raised from 1e-4 → 1e-2:
        #   The original 1e-4 is recommended for very deep networks (12+ blocks).
        #   With only 6 blocks the residual branches are less likely to destabilise,
        #   and the larger init gives gammas a stronger gradient signal from the start.
        model = SmallDataDecoderViT(
            in_channels=1,
            embed_dim=192,
            depth=6,
            num_heads=3,
            proj_drop=0.1,
            drop_path_rate=0.05,
            ls_init_value=1e-2,   # raised from 1e-4 → allows gammas to grow
        )

        model_path, fold_history = train_with_validation(
            model, train_loader, val_loader, fold, epochs=p.num_epochs
        )
        fold_models.append(model_path)
        all_fold_history.append(fold_history)

        # Explicitly release model, datasets, and cached CUDA memory between folds
        del model, train_loader, val_loader, train_dataset, val_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold += 1

    # Compare among the folds, choose the one with the highest val R^2 and/or lowest val MSE
    model_lowest_val_mse = np.argmin(
        [fold_history["val_mse"][-1] for fold_history in all_fold_history]
    )
    model_highest_val_r2 = np.argmax(
        [fold_history["val_r2"][-1].item() for fold_history in all_fold_history]
    )
    if model_lowest_val_mse == model_highest_val_r2:
        print("Lowest val_mse model = highest val_r2 model, all good.")
        print(
            f"Model {model_lowest_val_mse + 1} has the lowest val_MSE and the highest val_R^2."
        )
    else:
        print(
            "Lowest val_mse model != highest val_r2 model, choose lowest mse model."
        )
        print(
            f"Model {model_lowest_val_mse + 1} has the lowest val_MSE."
        )
        print(
            f"Model {model_highest_val_r2 + 1} has the highest val_R^2."
        )

    return fold_models[model_lowest_val_mse], all_fold_history

# To load model, define model class first.
#model = VisionForecaster(img_size=p.IMG_SIZE , patch_size=p.PATCH_SIZE, in_chans=p.CHANNELS, embed_dim=p.EMBED_DIM, depth=p.DEPTH, heads=p.HEADS, mlp_dim=p.MLP_DIM)
#PATH = 'your_model_path.pth' i.e., output of diff_model_multi_fold_cv_train_test() above.
#model.load_state_dict(torch.load(PATH))
