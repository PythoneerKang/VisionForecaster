import pickle
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import parameters as p
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformer import SmallDataDecoderViT

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GATE_WARMUP_EPOCHS  = 20
GATE_ENTROPY_WEIGHT = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

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


def _r2_from_scalars(ss_res: float, ss_tot: float) -> float:
    """Compute R² from running sum-of-squares accumulators."""
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


def _to_float(val) -> float:
    """Safely convert a tensor scalar or plain float to a Python float."""
    return val.item() if hasattr(val, "item") else float(val)


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Save the best model weights seen so far based on validation loss.

    The checkpoint written to `path` during training is a plain state_dict
    (weights only) — history is not available mid-training so it is not
    included here.  The full checkpoint with history is written at the end
    of train_with_validation() once training completes.
    """

    def __init__(self, patience=7, path='best_model.pt'):
        self.patience  = patience
        self.best_loss = float('inf')
        self.counter   = 0
        self.path      = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            # Save weights only — used internally to restore best weights
            # if early stopping fires before the final epoch.
            torch.save(_unwrap_state_dict(model), self.path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(model):
    """
    AdamW with FOUR parameter groups:

      1. decay_params   — weight matrices        (lr=1e-4, wd=1e-2)
      2. nodecay_params — biases, LN weights     (lr=1e-4, wd=0)
      3. gamma_params   — LayerScale gammas      (lr=1e-3, wd=0)   10× boost
      4. gate_params    — SectorGPSA gate_logit  (lr=1e-2, wd=0)  100× boost

    gate_logit gets a 100× LR boost because its gradient is doubly suppressed:
      - sigmoid derivative g*(1-g) ≈ 0.10 at gate_init=2
      - (v_pos - v_content) ≈ 0 early in training (both branches near-uniform)
    Without a LR boost the gate is effectively frozen throughout training.
    """
    decay_params   = []
    nodecay_params = []
    gamma_params   = []
    gate_params    = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gate_logit" in name:
            gate_params.append(param)
        elif "gamma" in name:
            gamma_params.append(param)
        elif param.ndim < 2:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    n_total   = sum(param.numel() for param in model.parameters() if param.requires_grad)
    n_grouped = sum(param.numel()
                    for group in [decay_params, nodecay_params, gamma_params, gate_params]
                    for param in group)
    assert n_total == n_grouped, (
        f"Parameter group mismatch: {n_total} total vs {n_grouped} grouped."
    )

    print(f"  Optimizer param groups:")
    print(f"    decay    : {sum(param.numel() for param in decay_params):>10,} params  lr=1e-4  wd=1e-2")
    print(f"    no-decay : {sum(param.numel() for param in nodecay_params):>10,} params  lr=1e-4  wd=0")
    print(f"    gamma    : {sum(param.numel() for param in gamma_params):>10,} params  lr=1e-3  wd=0   ← 10× boost")
    print(f"    gate     : {sum(param.numel() for param in gate_params):>10,} params  lr=1e-2  wd=0   ← 100× boost")

    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "lr": 1e-4, "weight_decay": 1e-2},
        {"params": nodecay_params, "lr": 1e-4, "weight_decay": 0.0},
        {"params": gamma_params,   "lr": 1e-3, "weight_decay": 0.0},
        {"params": gate_params,    "lr": 1e-2, "weight_decay": 0.0},
    ], lr=1e-4)

    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Gradient diagnostics
# ─────────────────────────────────────────────────────────────────────────────

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
    """Print gradient norms for all LayerScale gamma parameters."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Gate entropy regulariser
# ─────────────────────────────────────────────────────────────────────────────

def _gate_entropy_loss_fn(model) -> torch.Tensor:
    """
    Negative mean binary entropy of all gate values.

    Encourages gates to be neither all-0 nor all-1, nudging each head to
    find a meaningful blend of positional and content attention.

    Returns a scalar tensor on the same device as the model parameters,
    with gradient connectivity to gate_logit.
    """
    eps    = 1e-6
    device = next(model.parameters()).device
    terms  = []
    for name, param in model.named_parameters():
        if "gate_logit" in name:
            g = param.sigmoid()
            h = -(g * (g + eps).log() + (1.0 - g) * (1.0 - g + eps).log())
            terms.append(h.mean())
    if not terms:
        return torch.zeros(1, device=device)
    # Negative mean entropy (minimising this maximises entropy → pushes gates away from 0/1)
    return -torch.stack(terms).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Gate freeze / unfreeze
# ─────────────────────────────────────────────────────────────────────────────

def _set_gate_grad(model, requires_grad: bool):
    """Freeze or unfreeze all gate_logit parameters."""
    for name, param in model.named_parameters():
        if "gate_logit" in name:
            param.requires_grad_(requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (single fold)
# ─────────────────────────────────────────────────────────────────────────────

def train_with_validation(model, train_loader, val_loader, fold, epochs=100):

    device = torch.device("cuda" if p.USE_GPU and torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        model = torch.compile(model)
        print("  torch.compile: enabled")
    except Exception as e:
        print(f"  torch.compile: skipped ({e})")

    optimizer = _build_optimizer(model)
    criterion = nn.MSELoss()
    stopper   = EarlyStopping(patience=10, path=f'best_model_fold_{fold}.pt')

    # Freeze gates during warmup so the content stream has time to learn
    # meaningful QK representations before the gate is asked to balance them.
    _set_gate_grad(model, False)
    print(f"  Gate warmup: gate_logit FROZEN for first {GATE_WARMUP_EPOCHS} epochs")

    fold_history     = {'train_mse': [], 'val_mse': [], 'train_r2': [], 'val_r2': []}
    _grad_check_done = False

    for epoch in range(1, epochs + 1):
        print(f"----- Epoch {epoch} -----")

        # Unfreeze gates after warmup
        if epoch == GATE_WARMUP_EPOCHS + 1:
            _set_gate_grad(model, True)
            print(f"  Gate warmup complete: gate_logit UNFROZEN at epoch {epoch}")

        # ── TRAINING ──────────────────────────────────────────────────────
        model.train()
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

            mse_loss = criterion(outputs, y)

            # Gate entropy regulariser is only meaningful once gates are
            # unfrozen; during warmup gate_logit.requires_grad is False so
            # the regulariser has no effect on gate_logit (its value is still
            # computed but the backward graph stops at gate_logit).
            gate_reg   = _gate_entropy_loss_fn(model) * GATE_ENTROPY_WEIGHT
            total_loss = mse_loss + gate_reg

            total_loss.backward()

            if not _grad_check_done:
                _check_gamma_gradients(model)
                _check_gate_gradients(model)
                _grad_check_done = True

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track MSE only (not the reg term) so metrics stay comparable
            train_loss  += mse_loss.item()
            y_cpu        = y.detach().cpu().reshape(-1)
            out_cpu      = outputs.detach().cpu().reshape(-1)
            y_batches.append(y_cpu)
            pred_batches.append(out_cpu)
            train_sum_y += y_cpu.sum().item()
            train_n     += y_cpu.numel()

        y_all    = torch.cat(y_batches)
        p_all    = torch.cat(pred_batches)
        y_mean   = train_sum_y / train_n
        train_ss_res = ((p_all - y_all) ** 2).sum().item()
        train_ss_tot = ((y_all - y_mean) ** 2).sum().item()
        epoch_training_r2_score = _r2_from_scalars(train_ss_res, train_ss_tot)
        del y_batches, pred_batches, y_all, p_all

        # ── VALIDATION ────────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        val_n     = 0
        val_sum_y = 0.0
        vy_batches: list[torch.Tensor] = []
        vp_batches: list[torch.Tensor] = []
        print("Validation begins")

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y     = x.to(device), y.to(device)
                outputs   = model(x)
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

        # ── Gate and gamma monitoring ──────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1 or epoch == GATE_WARMUP_EPOCHS + 1:
            gate_vals = {
                name: torch.sigmoid(param).detach().cpu().mean().item()
                for name, param in model.named_parameters()
                if "gate_logit" in name
            }
            if gate_vals:
                mean_g = sum(gate_vals.values()) / len(gate_vals)
                min_g  = min(gate_vals.values())
                max_g  = max(gate_vals.values())
                frozen = epoch <= GATE_WARMUP_EPOCHS
                print(f"  Gate g=sigmoid(λ) — mean: {mean_g:.4f}  "
                      f"min: {min_g:.4f}  max: {max_g:.4f}  "
                      f"({'FROZEN' if frozen else 'trainable'})")

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
            getattr(model, '_orig_mod', model).load_state_dict(
                torch.load(stopper.path)
            )
            break

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


# ─────────────────────────────────────────────────────────────────────────────
# Multi-fold CV entry point
# ─────────────────────────────────────────────────────────────────────────────

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

    fold             = 1
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
