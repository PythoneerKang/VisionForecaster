import os
import pickle
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import parameters as p
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformer import SmallDataDecoderViT

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GATE_WARMUP_EPOCHS       = 2


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_state_dict(model):
    """
    Return the state_dict of the underlying module, stripping the
    '_orig_mod.' prefix that torch.compile() adds to all parameter keys.
    """
    return getattr(model, '_orig_mod', model).state_dict()


def _r2_from_scalars(ss_res: float, ss_tot: float) -> float:
    """Compute R² from running sum-of-squares accumulators.

    For delta-prediction the null model is ΔD=0 (no change), so ss_tot
    must be sum(y_scaled²) — NOT sum((y - ȳ)²).  Pass it accordingly.
    R² > 0 ⟺ model beats the zero-change null model.
    """
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

    The checkpoint saved at each improvement is the FULL dict (model weights +
    training history + scalers), not a bare state_dict.  This means there is
    no intermediate temporary file — the file at `path` is always a valid,
    self-contained checkpoint that can be loaded by test_one_fold.py.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before training is stopped.
    path : str
        File path for the checkpoint (e.g. "model_fold_3.pth").
    """

    def __init__(self, patience: int = 20, path: str = 'best_model.pt'):
        self.patience   = patience
        self.best_loss  = float('inf')
        self.best_epoch = 0
        self.counter    = 0
        self.path       = path

    def __call__(
        self,
        val_loss: float,
        model,
        epoch: int = 0,
        *,
        fold_history: dict | None = None,
        scaler_X_mean: float | None = None,
        scaler_X_std:  float | None = None,
        scaler_y_mean: float | None = None,
        scaler_y_std:  float | None = None,
    ) -> bool:
        """
        Evaluate val_loss.  If improved, save a full checkpoint and return False.
        After `patience` epochs without improvement, return True (stop training).
        """
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            # Build the full checkpoint dict so the saved file is always
            # self-contained — no separate scaler-injection step needed later.
            save_dict = {
                "model_state_dict": _unwrap_state_dict(model),
                "best_val_mse":     val_loss,
                "best_epoch":       epoch,
                "scaler_X_mean":    scaler_X_mean,
                "scaler_X_std":     scaler_X_std,
                "scaler_y_mean":    scaler_y_mean,
                "scaler_y_std":     scaler_y_std,
            }
            # Snapshot current history lists (shallow copy is sufficient —
            # the lists are rebuilt each epoch so old snapshots stay valid).
            if fold_history is not None:
                for k in ("train_mse", "val_mse", "train_r2", "val_r2"):
                    save_dict[k] = list(fold_history.get(k, []))
            torch.save(save_dict, self.path)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(model):
    """
    AdamW with FOUR parameter groups:

      1. decay_params   — weight matrices        (lr=1e-4, wd=1e-2)
      2. nodecay_params — biases, LN weights     (lr=1e-4, wd=0)
      3. gamma_params   — LayerScale gammas      (lr=1e-3, wd=0)   10× boost
      4. gate_params    — SectorGPSA gate_logit  (lr=1e-3, wd=0)   10× boost (reduced from 1e-2)

    gate_logit gets a 10× LR boost (1e-3 vs base 1e-4) to allow steady learning
    without the aggressive 100× boost that caused gate collapse. The warmup
    period (first 2 epochs) still sets gate LR to 0 to stabilize early training.
    """
    decay_params   = []
    nodecay_params = []
    gamma_params   = []
    gate_params    = []

    for name, param in model.named_parameters():
        # Include ALL parameters — do not filter by requires_grad
        if "gate_logit" in name:
            gate_params.append(param)
        elif "gamma" in name:
            gamma_params.append(param)
        elif param.ndim < 2:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    n_total   = sum(param.numel() for param in model.parameters())
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
    print(f"    gate     : {sum(param.numel() for param in gate_params):>10,} params  lr=1e-3  wd=0   ← 10× boost (reduced from 1e-2)")

    optimizer = torch.optim.AdamW([
        {"name": "decay",    "params": decay_params,   "lr": 1e-4, "weight_decay": 1e-2},
        {"name": "no-decay", "params": nodecay_params, "lr": 1e-4, "weight_decay": 0.0},
        {"name": "gamma",    "params": gamma_params,   "lr": 1e-3, "weight_decay": 0.0},
        # Gate group: base_lr=1e-3 (10× boost).  Frozen to 0 during warmup
        # via the GATE_WARMUP_EPOCHS block in train_with_validation().
        # Identified by name="gate" — never use a hardcoded index.
        {"name": "gate",     "params": gate_params,    "lr": 1e-3, "weight_decay": 0.0},
    ], lr=1e-4)

    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Gradient diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _check_gamma_gradients(model, epoch: int, batch_idx: int):
    """Print gradient norms for all LayerScale gamma parameters."""
    print(f"\n  ── LayerScale gamma gradient check (epoch {epoch}, batch {batch_idx}) ──")
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
# Training loop (single fold)
# ─────────────────────────────────────────────────────────────────────────────

def train_with_validation(
    model,
    train_loader,
    val_loader,
    fold,
    epochs=100,
    *,
    scaler_X_mean: float | None = None,
    scaler_X_std: float | None = None,
    scaler_y_mean: float | None = None,
    scaler_y_std: float | None = None,
    scaler_mean: float | None = None,   # backward compatibility
    scaler_std: float | None = None,    # backward compatibility
):
    """
    Train with validation using Cosine Annealing LR scheduler with warmup.

    Scheduler:
        - Warmup: first 5 epochs, LR increases linearly from 0 to base LR
        - Cosine decay: after warmup, LR follows cosine curve to 0
    The scheduler step() is called after each epoch (not after each batch).
    """
    device = torch.device("cuda" if p.USE_GPU and torch.cuda.is_available() else "cpu")
    model.to(device)

    # torch.compile is intentionally skipped: this codebase targets CPU-only
    # HPC nodes where compilation overhead exceeds any runtime benefit for
    # models of this size.  Re-enable with caution on GPU nodes.
    # try:
    #     model = torch.compile(model)
    # except Exception as e:
    #     print(f"  torch.compile: skipped ({e})")

    optimizer = _build_optimizer(model)

    # Fix #10: EarlyStopping saves directly to the final checkpoint path so
    # there is no intermediate bare-.pt file to manage or accidentally leave
    # behind if the process crashes between save and delete.
    model_path = f"model_fold_{fold}.pth"
    stopper    = EarlyStopping(patience=20, path=model_path)

    # Locate the gate param group by name — never by fragile index.
    # If _build_optimizer's group order changes, this still finds the right group.
    def _gate_group():
        for g in optimizer.param_groups:
            if g.get("name") == "gate":
                return g
        raise RuntimeError("Gate param group not found — check _build_optimizer().")

    # Store base LRs for each param group before any modifications.
    # This ensures warmup and cosine annealing use the correct base values.
    for param_group in optimizer.param_groups:
        param_group["base_lr"] = param_group["lr"]

    # ── Cosine annealing scheduler with warmup ─────────────────────────────
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    print(f"  LR scheduler: CosineAnnealingLR (warmup={warmup_epochs} epochs, T_max={epochs-warmup_epochs}, eta_min=1e-6)")

    fold_history     = {'train_mse': [], 'val_mse': [], 'train_r2': [], 'val_r2': []}
    _grad_check_done = False

    # Fix #1: compute the correct null prediction in scaled space once per fold.
    # The null model predicts ΔD=0 in raw space, which in scaled space is
    #   ŷ_null = (0 − y_mean) / y_std = −y_mean / y_std
    # SS_null must use this constant, not 0, unless y_mean happens to be 0.
    # scaler_y_mean / scaler_y_std may be None for legacy callers — fall back to 0.
    if scaler_y_mean is not None and scaler_y_std is not None and float(scaler_y_std) != 0.0:
        null_pred_scaled = float(-scaler_y_mean) / float(scaler_y_std)
    else:
        null_pred_scaled = 0.0   # legacy / unknown scaler: assume y_mean≈0

    for epoch in range(1, epochs + 1):
        print(f"----- Epoch {epoch} -----")

        # ── LR scheduling ────────────────────────────────────────────────
        # Two schedules interact each epoch:
        #
        # 1. Gate warmup (epochs 1–GATE_WARMUP_EPOCHS):
        #    Gate LR is held at exactly 0 so the sector-positional prior is
        #    stable while all other parameters initialise.
        #
        # 2. Cosine warmup for all other params (epochs 1–warmup_epochs):
        #    LR rises linearly from 0 to base_lr, then cosine-decays.
        #
        # Interaction note: at epoch GATE_WARMUP_EPOCHS+1 (epoch 3), if we
        # are still inside the cosine warmup window (epoch <= warmup_epochs=5),
        # the cosine warmup block below will scale the restored gate LR by
        # (epoch/warmup_epochs).  This is intentional — the gate LR ramps up
        # together with all other LRs during the cosine warmup window.
        # The print statement reflects the final LR set this epoch, not an
        # intermediate value.

        gate_grp = _gate_group()

        if epoch <= GATE_WARMUP_EPOCHS:
            gate_grp["lr"] = 0.0
        elif epoch == GATE_WARMUP_EPOCHS + 1:
            # Restore gate LR to base; cosine warmup below may scale it further.
            gate_grp["lr"] = gate_grp["base_lr"]

        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                if param_group.get("name") == "gate" and epoch <= GATE_WARMUP_EPOCHS:
                    continue   # keep gate frozen — don't overwrite the 0
                param_group["lr"] = param_group["base_lr"] * warmup_factor
            if epoch == GATE_WARMUP_EPOCHS + 1:
                print(f"  Gate warmup complete: gate_logit LR set to "
                      f"{gate_grp['lr']:.2e} at epoch {epoch} "
                      f"(warmup factor {warmup_factor:.2f} applied)")
        else:
            scheduler.step()
            if epoch == GATE_WARMUP_EPOCHS + 1:
                print(f"  Gate warmup complete: gate_logit LR restored to "
                      f"{gate_grp['lr']:.2e} at epoch {epoch}")

        # ── TRAINING ──────────────────────────────────────────────────────
        model.train()
        train_sse = 0.0
        train_n   = 0
        y_batches:    list[torch.Tensor] = []
        pred_batches: list[torch.Tensor] = []
        print("Training begins")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            x, y = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(x)

            # MSE loss vs the scaled ΔD target.
            # The null-model MSE (predict ΔD=0 raw, i.e. −y_mean/y_std scaled)
            # equals mean(y_scaled²) ≈ 1.0 on the training set by construction.
            # The model must push val MSE below 1.0 to beat the null model.
            total_loss = F.mse_loss(outputs, y)

            total_loss.backward()

            if not _grad_check_done:
                _check_gamma_gradients(model, epoch, batch_idx)
                _grad_check_done = True

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_sse += F.mse_loss(outputs, y, reduction="sum").item()
            y_cpu      = y.detach().cpu().reshape(-1)
            out_cpu    = outputs.detach().cpu().reshape(-1)
            y_batches.append(y_cpu)
            pred_batches.append(out_cpu)
            train_n   += y_cpu.numel()

        y_all    = torch.cat(y_batches)
        p_all    = torch.cat(pred_batches)

        # R² vs zero-change null model (null: predict ΔD=0 in raw space).
        # In scaled space the null predicts null_pred_scaled = −y_mean/y_std,
        # so SS_null = Σ (y_scaled − null_pred_scaled)².
        # R² > 0  ⟺  model beats the zero-change null model.
        train_ss_res  = ((p_all - y_all) ** 2).sum().item()
        train_ss_null = ((y_all - null_pred_scaled) ** 2).sum().item()
        epoch_training_r2_score = _r2_from_scalars(train_ss_res, train_ss_null)
        train_mean_abs_pred = p_all.abs().mean().item()
        train_mean_abs_y    = y_all.abs().mean().item()
        del y_batches, pred_batches, y_all, p_all

        # ── VALIDATION ────────────────────────────────────────────────────
        model.eval()
        val_sse  = 0.0
        val_n    = 0
        vy_batches: list[torch.Tensor] = []
        vp_batches: list[torch.Tensor] = []
        print("Validation begins")

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y     = x.to(device), y.to(device)
                outputs   = model(x)
                val_sse  += F.mse_loss(outputs, y, reduction="sum").item()
                y_cpu    = y.detach().cpu().reshape(-1)
                out_cpu  = outputs.detach().cpu().reshape(-1)
                vy_batches.append(y_cpu)
                vp_batches.append(out_cpu)
                val_n    += y_cpu.numel()

        vy_all  = torch.cat(vy_batches)
        vp_all  = torch.cat(vp_batches)
        # R² vs zero-change null model (same definition as training)
        val_ss_res  = ((vp_all - vy_all) ** 2).sum().item()
        val_ss_null = ((vy_all - null_pred_scaled) ** 2).sum().item()
        epoch_validation_r2_score = _r2_from_scalars(val_ss_res, val_ss_null)
        val_mean_abs_pred = vp_all.abs().mean().item()
        val_mean_abs_y    = vy_all.abs().mean().item()
        del vy_batches, vp_batches, vy_all, vp_all

        avg_train = train_sse / train_n if train_n > 0 else 0.0
        avg_val   = val_sse   / val_n   if val_n   > 0 else 0.0
        # Null-model MSE in scaled space: mean((y_scaled − null_pred_scaled)²)
        null_mse_approx = val_ss_null / val_n if val_n > 0 else 1.0
        print(f"----- Train/Validation results -----")
        print(f"Epoch {epoch}: Train MSE {avg_train:.6f} | Val MSE {avg_val:.6f} | "
              f"Null MSE (val) {null_mse_approx:.6f}")
        print(f"Epoch {epoch}: Train R²(vs null): {epoch_training_r2_score * 100:.4f}% | "
              f"Val R²(vs null): {epoch_validation_r2_score * 100:.4f}%  "
              f"[>0 ⟺ beats ΔD=0 null model]")
        print(f"Epoch {epoch}: MeanAbs | Train pred={train_mean_abs_pred:.6f} y={train_mean_abs_y:.6f} | "
              f"Val pred={val_mean_abs_pred:.6f} y={val_mean_abs_y:.6f}")

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
                frozen   = epoch <= GATE_WARMUP_EPOCHS
                gate_lr  = _gate_group()["lr"]
                print(f"  Gate g=sigmoid(λ) — mean: {mean_g:.4f}  "
                      f"min: {min_g:.4f}  max: {max_g:.4f}  "
                      f"({'FROZEN lr=0' if frozen else f'trainable lr={gate_lr:.2e}'})")

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

        if stopper(
            avg_val, model, epoch,
            fold_history=fold_history,
            scaler_X_mean=scaler_X_mean,
            scaler_X_std=scaler_X_std,
            scaler_y_mean=scaler_y_mean,
            scaler_y_std=scaler_y_std,
        ):
            print("Early stopping triggered. Loading best model weights...")
            best_ckpt = torch.load(stopper.path, weights_only=False)
            getattr(model, '_orig_mod', model).load_state_dict(
                best_ckpt["model_state_dict"]
            )
            print(f"  Restored weights from best epoch {stopper.best_epoch} "
                  f"(val_loss={stopper.best_loss:.6f})")
            break

    # Print final learning rates for all groups
    print("\n  Final learning rates:")
    for group in optimizer.param_groups:
        print(f"    {group.get('name', '?'):10s}: lr={group['lr']:.6f}")

    # The final checkpoint was already saved by EarlyStopping (or on the last
    # epoch if training ran to completion without early stopping).  Update it
    # now to include the complete history and best_val_mse.
    # If training ended naturally (no early stopping), save the final state.
    if not os.path.isfile(model_path):
        # Training completed all epochs without early stopping — save now.
        save_dict = {
            "model_state_dict": _unwrap_state_dict(model),
            "best_val_mse":     stopper.best_loss,
            "best_epoch":       stopper.best_epoch,
            "scaler_X_mean":    scaler_X_mean,
            "scaler_X_std":     scaler_X_std,
            "scaler_y_mean":    scaler_y_mean,
            "scaler_y_std":     scaler_y_std,
            "train_mse":        fold_history["train_mse"],
            "val_mse":          fold_history["val_mse"],
            "train_r2":         fold_history["train_r2"],
            "val_r2":           fold_history["val_r2"],
        }
        torch.save(save_dict, model_path)
    else:
        # EarlyStopping already saved the best-epoch checkpoint.  Overwrite it
        # with the full history (EarlyStopping snapshots history up to the best
        # epoch; we now have the complete run history to store).
        existing = torch.load(model_path, weights_only=False)
        existing["train_mse"] = fold_history["train_mse"]
        existing["val_mse"]   = fold_history["val_mse"]
        existing["train_r2"]  = fold_history["train_r2"]
        existing["val_r2"]    = fold_history["val_r2"]
        torch.save(existing, model_path)

    fold_history["best_val_mse"] = stopper.best_loss
    best_epoch_msg = f"  best_epoch={stopper.best_epoch}"
    print(f"  Checkpoint saved → {model_path}  (weights + history){best_epoch_msg}")

    return model_path, fold_history


# ─────────────────────────────────────────────────────────────────────────────
# Multi-fold CV entry point
# ─────────────────────────────────────────────────────────────────────────────

def diff_model_multi_fold_cv_train_test(
    distance_matrix: np.ndarray,
    sector_ids: torch.Tensor,
    model_cfg: dict,
):
    """
    Perform multi-fold CV training using SectorGPSA-based SmallDataDecoderViT.

    The model predicts the *change* in distance matrices:
        y = D_{t+1} - D_t

    Parameters
    ----------
    distance_matrix : (T, 457, 457) float32 array — GICS-reordered.
    sector_ids      : (N,) long tensor — patch→sector mapping.
    model_cfg       : dict — SmallDataDecoderViT kwargs (excluding sector_ids).
                      Pass MODEL_CFG from main.py so training and interpretability
                      always use the identical architecture.
    """
    if p.TORCH_NUM_THREADS is not None:
        torch.set_num_threads(p.TORCH_NUM_THREADS)
    if p.TORCH_NUM_INTEROP_THREADS is not None:
        torch.set_num_interop_threads(p.TORCH_NUM_INTEROP_THREADS)

    X = distance_matrix[:-1][:, np.newaxis, :]  # D_t
    y = (distance_matrix[1:] - distance_matrix[:-1])[:, np.newaxis, :]  # ΔD

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)

    fold             = 1
    fold_models      = []
    all_fold_history = []

    for train_index, val_index in tscv.split(X_tensor):
        print(10 * "=", f"Fold={fold}", 10 * "=")

        X_train_raw, X_val_raw = X_tensor[train_index], X_tensor[val_index]
        y_train_raw, y_val_raw = y_tensor[train_index], y_tensor[val_index]

        # Fold-wise standardisation (train-only)
        X_mean = X_train_raw.mean().item()
        X_std  = X_train_raw.std(unbiased=False).item()
        y_mean = y_train_raw.mean().item()
        y_std  = y_train_raw.std(unbiased=False).item()

        if X_std == 0.0 or y_std == 0.0:
            raise ValueError(f"Fold {fold}: standard deviation is zero; cannot z-score.")

        X_train = (X_train_raw - X_mean) / X_std
        y_train = (y_train_raw - y_mean) / y_std
        X_val   = (X_val_raw   - X_mean) / X_std
        y_val   = (y_val_raw   - y_mean) / y_std

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset   = TensorDataset(X_val,   y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=p.BATCH_SIZE, shuffle=True,
            num_workers=p.NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=p.BATCH_SIZE, shuffle=False,
            num_workers=p.NUM_WORKERS, pin_memory=True,
        )

        model = SmallDataDecoderViT(
            **model_cfg,
            sector_ids=sector_ids,
        )

        model_path, fold_history = train_with_validation(
            model,
            train_loader,
            val_loader,
            fold,
            epochs=p.num_epochs,
            scaler_X_mean=X_mean,
            scaler_X_std=X_std,
            scaler_y_mean=y_mean,
            scaler_y_std=y_std,
        )
        fold_models.append(model_path)
        all_fold_history.append(fold_history)

        del model, train_loader, val_loader, train_dataset, val_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold += 1

    # Use best_val_mse (the val MSE at the checkpoint epoch) for fold selection,
    # not fh["val_mse"][-1] (the last epoch before stopping, which is always
    # worse than the best because early stopping waited patience epochs).
    final_val_mse = [fh["best_val_mse"] for fh in all_fold_history]
    # For R², use the val_r2 at the best epoch index
    final_val_r2 = []
    for fh in all_fold_history:
        best_ep_idx = int(np.argmin(fh["val_mse"]))   # 0-indexed
        final_val_r2.append(_to_float(fh["val_r2"][best_ep_idx]))

    print("\n" + "=" * 60)
    print("FOLD SUMMARY — null model: predict ΔD=0 (no change)")
    print(f"  Null-model val MSE ≈ 1.0 in scaled space (by construction).")
    print(f"  Model beats null iff val MSE < null MSE, i.e. R²(vs null) > 0.")
    print(f"  {'Fold':>5}  {'Val MSE':>12}  {'R²(vs null)':>14}  {'Beats null?':>12}")
    print("  " + "-" * 50)
    for i, (mse, r2) in enumerate(zip(final_val_mse, final_val_r2)):
        beats = "✓" if r2 > 0 else "✗"
        print(f"  {i+1:>5}  {mse:>12.6f}  {r2*100:>13.4f}%  {beats:>12}")
    print("=" * 60 + "\n")

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