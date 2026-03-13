import pickle
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import parameters as p
import torch
from transformer import *
from torch.utils.data import DataLoader, TensorDataset


def _unwrap_state_dict(model):
    """
    Return the state_dict of the underlying module, stripping the
    '_orig_mod.' prefix that torch.compile() adds to all parameter keys.

    Without this, a checkpoint saved from a compiled model cannot be loaded
    into an uncompiled SmallDataDecoderViT because every key differs:
        compiled   : '_orig_mod.pos_embed', '_orig_mod.blocks.0.attn.gate_logit', …
        uncompiled : 'pos_embed',           'blocks.0.attn.gate_logit', …

    torch.compile wraps the module in an OptimizedModule that exposes the
    original model via the '_orig_mod' attribute.  Falling back to `model`
    itself handles the case where compile was skipped (older PyTorch build).
    """
    return getattr(model, '_orig_mod', model).state_dict()


class EarlyStopping:
    """
    Save the best model weights seen so far based on validation loss.

    The checkpoint written to `path` during training is a plain state_dict
    (weights only) — history is not available mid-training so it is not
    included here.  The full checkpoint with history is written at the end
    of train_with_validation() once training completes.
    """

    def __init__(self, patience=7, path='best_model.pt'):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Save weights only — used internally to restore best weights
            # if early stopping fires before the final epoch.
            torch.save(_unwrap_state_dict(model), self.path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def _build_optimizer(model):
    """
    Build an AdamW optimizer with three parameter groups:

      1. decay_params   — weight matrices (lr=1e-4, weight_decay=1e-2)
      2. nodecay_params — biases, LayerNorm weights, gate logits (lr=1e-4, wd=0)
      3. gamma_params   — LayerScale gammas (lr=1e-3, wd=0)
                          10× higher LR so gammas escape their near-zero init.

    gate_logit parameters are 1-D scalars and fall into the nodecay group
    automatically (param.ndim < 2).  They do not need a boosted LR because
    sigmoid is well-conditioned and their init (λ=+2) already places them in
    a region with meaningful gradient.
    """
    decay_params, nodecay_params, gamma_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gamma" in name:
            gamma_params.append(param)
        elif param.ndim < 2:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    n_total   = sum(param_.numel() for param_ in model.parameters() if param_.requires_grad)
    n_grouped = (sum(param_.numel() for param_ in decay_params) +
                 sum(param_.numel() for param_ in nodecay_params) +
                 sum(param_.numel() for param_ in gamma_params))
    assert n_total == n_grouped, (
        f"Parameter group mismatch: {n_total} total vs {n_grouped} grouped."
    )

    print(f"  Optimizer param groups:")
    print(f"    decay    : {sum(param_.numel() for param_ in decay_params):>10,} params  lr=1e-4  wd=1e-2")
    print(f"    no-decay : {sum(param_.numel() for param_ in nodecay_params):>10,} params  lr=1e-4  wd=0  (incl. gate_logit)")
    print(f"    gamma    : {sum(param_.numel() for param_ in gamma_params):>10,} params  lr=1e-3  wd=0  ← 10× boost")

    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "lr": 1e-4, "weight_decay": 1e-2},
        {"params": nodecay_params, "lr": 1e-4, "weight_decay": 0.0},
        {"params": gamma_params,   "lr": 1e-3, "weight_decay": 0.0},
    ], lr=1e-4)

    return optimizer


def _check_gate_gradients(model):
    """
    Print gradient norms for all SectorGPSA gate_logit parameters.
    Call once after the first backward pass to verify gates are receiving
    meaningful gradients.

    NOTE: gate_logit grad norms are expected to be extremely small (~0) at
    epoch 1 batch 1.  With gate_init=+2.0, sigmoid(λ)≈0.88, meaning the
    positional branch dominates and the gate gradient is:
        dL/dλ = dL/dg · g·(1-g)  ≈  dL/dg · 0.105
    This near-zero reading does NOT indicate a dead parameter — the gate
    will accumulate gradient signal over many batches as the content branch
    diverges from the positional prior.  Monitor gate values (g=sigmoid(λ))
    across epochs instead of this single-batch gradient check.
    """
    print("\n  ── SectorGPSA gate_logit gradient check (epoch 1, batch 1) ──")
    print("  (near-zero grads are expected at init; see docstring)")
    any_printed = False
    for name, param in model.named_parameters():
        if "gate_logit" in name:
            if param.grad is not None:
                print(f"    {name:50s}  grad norm = {param.grad.norm().item():.6f}")
            else:
                print(f"    {name:50s}  grad = None  ← not connected!")
            any_printed = True
    if not any_printed:
        print("    WARNING: no gate_logit parameters found in model!")
    print("  ── end gate check ──\n")


def _check_gamma_gradients(model):
    """
    Print gradient norms for all LayerScale gamma parameters.
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


def _r2_from_scalars(ss_res: float, ss_tot: float) -> float:
    """Compute R² from running sum-of-squares accumulators."""
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


def _to_float(val) -> float:
    """Safely convert a tensor scalar or plain float to a Python float."""
    return val.item() if hasattr(val, "item") else float(val)


def train_with_validation(model, train_loader, val_loader, fold, epochs=100):

    device = torch.device("cuda" if p.USE_GPU and torch.cuda.is_available() else "cpu")
    model.to(device)

    # torch.compile fuses elementwise ops (gate interpolation, LayerScale,
    # residuals) into optimised kernels.  Falls back gracefully on older
    # PyTorch builds via the try/except.
    try:
        model = torch.compile(model)
        print("  torch.compile: enabled")
    except Exception as e:
        print(f"  torch.compile: skipped ({e})")

    optimizer = _build_optimizer(model)
    criterion = nn.MSELoss()
    stopper   = EarlyStopping(patience=10)

    fold_history = {'train_mse': [], 'val_mse': [], 'train_r2': [], 'val_r2': []}

    _grad_check_done = False

    for epoch in range(1, epochs + 1):
        print(f"----- Epoch {epoch} -----")

        # --- TRAINING PHASE ---
        model.train()
        # Incremental R² accumulators — avoids storing all predictions/targets
        # in memory (~460 MB of CPU tensors per epoch for the 504-sample
        # training set at 464×464 float32).
        # We keep flattened 1-D CPU copies (much cheaper than 4-D tensors)
        # only until the end of the epoch, then immediately discard them.
        train_loss  = 0.0
        train_n     = 0
        train_sum_y = 0.0
        y_batches:    list[torch.Tensor] = []
        pred_batches: list[torch.Tensor] = []
        print("Training begins")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            x, y = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss    = criterion(outputs, y)
            loss.backward()

            if not _grad_check_done:
                _check_gamma_gradients(model)
                _check_gate_gradients(model)
                _grad_check_done = True

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss  += loss.item()
            y_cpu        = y.detach().cpu().reshape(-1)
            out_cpu      = outputs.detach().cpu().reshape(-1)
            y_batches.append(y_cpu)
            pred_batches.append(out_cpu)
            train_sum_y += y_cpu.sum().item()
            train_n     += y_cpu.numel()

        # Single-pass R² over already-CPU 1-D tensors
        y_all   = torch.cat(y_batches)
        p_all   = torch.cat(pred_batches)
        y_mean  = train_sum_y / train_n
        train_ss_res = ((p_all - y_all) ** 2).sum().item()
        train_ss_tot = ((y_all - y_mean) ** 2).sum().item()
        epoch_training_r2_score = _r2_from_scalars(train_ss_res, train_ss_tot)
        del y_batches, pred_batches, y_all, p_all

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss  = 0.0
        val_n     = 0
        val_sum_y = 0.0
        vy_batches: list[torch.Tensor] = []
        vp_batches: list[torch.Tensor] = []
        print("Validation begins")

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y    = x.to(device), y.to(device)
                outputs  = model(x)
                val_loss += criterion(outputs, y).item()
                y_cpu    = y.detach().cpu().reshape(-1)
                out_cpu  = outputs.detach().cpu().reshape(-1)
                vy_batches.append(y_cpu)
                vp_batches.append(out_cpu)
                val_sum_y += y_cpu.sum().item()
                val_n     += y_cpu.numel()

        vy_all  = torch.cat(vy_batches)
        vp_all  = torch.cat(vp_batches)
        vy_mean = val_sum_y / val_n
        val_ss_res = ((vp_all - vy_all) ** 2).sum().item()
        val_ss_tot = ((vy_all - vy_mean) ** 2).sum().item()
        epoch_validation_r2_score = _r2_from_scalars(val_ss_res, val_ss_tot)
        del vy_batches, vp_batches, vy_all, vp_all

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        print(f"----- Train/Validation results -----")
        print(f"Epoch {epoch}: Train Loss {avg_train:.6f} | Val Loss {avg_val:.6f}")
        print(f"Epoch {epoch}: Train R^2: {epoch_training_r2_score * 100:.4f}% | "
              f"Val R^2: {epoch_validation_r2_score * 100:.4f}%")

        # ── Log gate values and gamma values periodically ─────────────────
        if epoch % 10 == 0 or epoch == 1:
            # Gate values (g = sigmoid(lambda))
            gate_vals = {
                name: torch.sigmoid(param).detach().cpu().mean().item()
                for name, param in model.named_parameters()
                if "gate_logit" in name
            }
            if gate_vals:
                mean_g = sum(gate_vals.values()) / len(gate_vals)
                min_g  = min(gate_vals.values())
                max_g  = max(gate_vals.values())
                print(f"  Gate g=sigmoid(λ) — mean: {mean_g:.4f}  "
                      f"min: {min_g:.4f}  max: {max_g:.4f}  "
                      f"(0=content, 1=positional)")

            # LayerScale gammas
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

        fold_history['train_mse'].append(avg_train)
        fold_history['val_mse'].append(avg_val)
        fold_history['train_r2'].append(epoch_training_r2_score)
        fold_history['val_r2'].append(epoch_validation_r2_score)

        if stopper(avg_val, model):
            print("Early stopping triggered. Loading best model weights...")
            # FIX: load into the unwrapped model so the bare keys in the
            # checkpoint match the uncompiled module's parameter names.
            getattr(model, '_orig_mod', model).load_state_dict(
                torch.load(stopper.path)
            )
            break

    # Save a single checkpoint containing both the model weights and the
    # full training history.  scratch.py reads this one file for everything —
    # no separate .pkl needed.
    #
    # Checkpoint schema
    # -----------------
    # {
    #   "model_state_dict" : bare state_dict (no _orig_mod. prefix),
    #   "train_mse"        : list[float] — per-epoch training MSE,
    #   "val_mse"          : list[float] — per-epoch validation MSE,
    #   "train_r2"         : list[float] — per-epoch training R²,
    #   "val_r2"           : list[float] — per-epoch validation R²,
    # }
    model_path = f"model_fold_{fold}.pth"
    torch.save(
        {
            "model_state_dict": _unwrap_state_dict(model),
            "train_mse":        fold_history["train_mse"],
            "val_mse":          fold_history["val_mse"],
            "train_r2":         fold_history["train_r2"],
            "val_r2":           fold_history["val_r2"],
        },
        model_path,
    )
    print(f"  Checkpoint saved → {model_path}  (weights + history)")

    return model_path, fold_history


def diff_model_multi_fold_cv_train_test(
    distance_matrix: np.ndarray,
    sector_ids: torch.Tensor,
):
    """
    Perform multi-fold CV training using SectorGPSA-based SmallDataDecoderViT.

    Parameters
    ----------
    distance_matrix : np.ndarray  shape (T, 457, 457)
        GICS-reordered, z-scored distance matrices.
    sector_ids : torch.Tensor  shape (N,) dtype=torch.long
        Patch-level GICS sector indices, from build_patch_sector_ids().
    """
    if p.TORCH_NUM_THREADS is not None:
        torch.set_num_threads(p.TORCH_NUM_THREADS)
    if p.TORCH_NUM_INTEROP_THREADS is not None:
        torch.set_num_interop_threads(p.TORCH_NUM_INTEROP_THREADS)

    X = distance_matrix[:-1][:, np.newaxis, :]
    y = distance_matrix[1:][:, np.newaxis, :]

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)

    fold = 1
    fold_models      = []
    all_fold_history = []

    for train_index, val_index in tscv.split(X_tensor):
        print(10 * "=", f"Fold={fold}", 10 * "=")

        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset   = TensorDataset(X_val,   y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=p.BATCH_SIZE, shuffle=False,
            num_workers=p.NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=p.BATCH_SIZE, shuffle=False,
            num_workers=p.NUM_WORKERS, pin_memory=True,
        )

        # Sector-GPSA model: sector_ids drives the positional attention prior.
        # gate_init=2.0 → sigmoid(2) ≈ 0.88 positional at start of training.
        model = SmallDataDecoderViT(
            in_channels=1,
            embed_dim=192,
            depth=6,
            num_heads=3,
            proj_drop=0.1,
            drop_path_rate=0.05,
            ls_init_value=1e-2,
            gate_init=2.0,
            sector_ids=sector_ids,
        )

        model_path, fold_history = train_with_validation(
            model, train_loader, val_loader, fold, epochs=p.num_epochs
        )
        fold_models.append(model_path)
        all_fold_history.append(fold_history)

        del model, train_loader, val_loader, train_dataset, val_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold += 1

    # _r2_from_scalars returns a plain Python float, so use _to_float()
    # rather than .item() to safely handle both tensor scalars and floats.
    final_val_mse = [fh["val_mse"][-1] for fh in all_fold_history]
    final_val_r2  = [_to_float(fh["val_r2"][-1]) for fh in all_fold_history]

    model_lowest_val_mse = int(np.argmin(final_val_mse))
    model_highest_val_r2 = int(np.argmax(final_val_r2))

    if model_lowest_val_mse == model_highest_val_r2:
        print("Lowest val_mse model = highest val_r2 model, all good.")
        print(f"Model {model_lowest_val_mse + 1} has the lowest val_MSE and highest val_R^2.")
    else:
        print("Lowest val_mse model != highest val_r2 model, choosing lowest mse model.")
        print(f"Model {model_lowest_val_mse + 1} has the lowest val_MSE.")
        print(f"Model {model_highest_val_r2 + 1} has the highest val_R^2.")

    return fold_models[model_lowest_val_mse], all_fold_history
