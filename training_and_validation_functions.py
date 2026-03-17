"""
Memory-Optimized Training Functions
====================================

OPTIMIZATIONS:
1. Gradient accumulation - simulate larger batches with less memory
2. Mixed precision training (AMP) - use FP16 for reduced memory
3. Efficient batch processing
4. Memory cleanup between batches
5. Optimized validation loop
"""

import parameters as p
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from transformer import SmallDataDecoderViT
from pathlib import Path
import gc


def _r2_from_scalars(ss_res: float, ss_tot: float, eps: float = 1e-12) -> float:
    """Compute R² score from sum of squares."""
    return 1.0 - ss_res / max(ss_tot, eps)


def _check_gamma_gradients(model: torch.nn.Module, epoch: int, batch: int):
    """Debug helper for LayerScale gradients."""
    for i, blk in enumerate(model.blocks):
        g1 = blk.ls1.gamma.grad
        g2 = blk.ls2.gamma.grad
        if g1 is not None and g2 is not None:
            print(f"  Block {i}: γ₁.grad={g1.abs().mean().item():.2e}, "
                  f"γ₂.grad={g2.abs().mean().item():.2e}")


def train_with_validation_optimized(
    model: SmallDataDecoderViT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold: int,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gate_lr_mult: float = 0.05,
    warmup_epochs: int = 5,
    scaler_X_mean: float = 0.0,
    scaler_X_std: float = 1.0,
    scaler_y_mean: float = 0.0,
    scaler_y_std: float = 1.0,
    gradient_accumulation_steps: int = 2,
    use_amp: bool = False,
    memory_efficient_validation: bool = True,
    early_stopping: bool = False,  # NEW: enable early stopping
    early_stopping_patience: int = 15,  # NEW: patience for early stopping
    gate_warmup_epochs: int = 10,  # NEW: configurable gate warmup
):
    """
    Memory-optimized training with:
    - Gradient accumulation (simulate larger batches)
    - Mixed precision training (FP16)
    - Memory-efficient validation
    - Aggressive memory cleanup
    """
    device = torch.device("cuda" if torch.cuda.is_available() and p.USE_GPU else "cpu")
    model = model.to(device)
    
    # Null model prediction in scaled space
    null_pred_scaled = -scaler_y_mean / scaler_y_std if scaler_y_std != 0.0 else 0.0
    
    # Optimizer with separate parameter groups
    attn_params = []
    gate_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gate_logit" in name:
            gate_params.append(param)
        elif "attn" in name or "qkv" in name:
            attn_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": other_params, "name": "other", "base_lr": lr},
        {"params": attn_params, "name": "attn", "base_lr": lr},
        {"params": gate_params, "name": "gate", "base_lr": lr * gate_lr_mult},
    ], lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Training history
    history = {
        "train_loss": [], "train_r2": [],
        "val_loss": [], "val_r2": [],
        "train_mae": [], "val_mae": [],
        "lr": [], "gate_lr": []
    }
    
    best_val_r2 = -float("inf")
    best_model_path = Path(f"best_model_fold_{fold}.pt")
    patience_counter = 0  # NEW: for early stopping
    
    GATE_WARMUP_EPOCHS = gate_warmup_epochs
    _grad_check_done = False
    
    print(f"\n{'='*70}")
    print(f"MEMORY-OPTIMIZED TRAINING - Fold {fold}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {p.BATCH_SIZE * gradient_accumulation_steps}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Gradient checkpointing: {model.use_checkpoint}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, epochs + 1):
        # Gate warmup
        gate_grp = optimizer.param_groups[2]
        if epoch <= GATE_WARMUP_EPOCHS:
            gate_grp["lr"] = 0.0
        elif epoch == GATE_WARMUP_EPOCHS + 1:
            gate_grp["lr"] = gate_grp["base_lr"]
        
        # LR warmup
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                if param_group.get("name") == "gate" and epoch <= GATE_WARMUP_EPOCHS:
                    continue
                param_group["lr"] = param_group["base_lr"] * warmup_factor
        else:
            scheduler.step()
        
        # ── TRAINING (with gradient accumulation) ─────────────────────────
        model.train()
        train_loss = 0.0
        train_n = 0
        y_batches = []
        pred_batches = []
        
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            x, y = inputs.to(device), labels.to(device)
            
            # Mixed precision forward pass
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(x)
                    loss = F.mse_loss(outputs, y)
                    loss = loss / gradient_accumulation_steps  # Scale loss
            else:
                outputs = model(x)
                loss = F.mse_loss(outputs, y)
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step (every N accumulation steps)
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if not _grad_check_done:
                    _check_gamma_gradients(model, epoch, batch_idx)
                    _grad_check_done = True
                
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Accumulate metrics (use unscaled outputs/labels, not the divided loss)
            with torch.no_grad():
                batch_mse = F.mse_loss(outputs.detach(), y.detach(), reduction="sum").item()
            train_loss += batch_mse
            y_cpu = y.detach().cpu().reshape(-1)
            out_cpu = outputs.detach().cpu().reshape(-1)
            y_batches.append(y_cpu)
            pred_batches.append(out_cpu)
            train_n += y_cpu.numel()
            
            # Memory cleanup
            del x, y, outputs, loss
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Handle remaining gradients
        if (len(train_loader) % gradient_accumulation_steps) != 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        # Compute training metrics
        y_all = torch.cat(y_batches)
        p_all = torch.cat(pred_batches)
        train_ss_res = ((p_all - y_all) ** 2).sum().item()
        train_ss_null = ((y_all - null_pred_scaled) ** 2).sum().item()
        train_r2 = _r2_from_scalars(train_ss_res, train_ss_null)
        train_mae = (p_all - y_all).abs().mean().item()
        
        del y_batches, pred_batches, y_all, p_all
        gc.collect()
        
        # ── VALIDATION (memory-efficient) ──────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_n = 0
        val_y_batches = []
        val_pred_batches = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                x, y = inputs.to(device), labels.to(device)
                
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(x)
                else:
                    outputs = model(x)
                
                val_loss += F.mse_loss(outputs, y, reduction="sum").item()
                y_cpu = y.detach().cpu().reshape(-1)
                out_cpu = outputs.detach().cpu().reshape(-1)
                val_y_batches.append(y_cpu)
                val_pred_batches.append(out_cpu)
                val_n += y_cpu.numel()
                
                del x, y, outputs
        
        val_y_all = torch.cat(val_y_batches)
        val_p_all = torch.cat(val_pred_batches)
        val_ss_res = ((val_p_all - val_y_all) ** 2).sum().item()
        val_ss_null = ((val_y_all - null_pred_scaled) ** 2).sum().item()
        val_r2 = _r2_from_scalars(val_ss_res, val_ss_null)
        val_mae = (val_p_all - val_y_all).abs().mean().item()
        
        del val_y_batches, val_pred_batches, val_y_all, val_p_all
        gc.collect()
        
        # Update history
        history["train_loss"].append(train_loss / train_n)
        history["train_r2"].append(train_r2)
        history["val_loss"].append(val_loss / val_n)
        history["val_r2"].append(val_r2)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["gate_lr"].append(optimizer.param_groups[2]["lr"])
        
        # Save best model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0  # Reset patience
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_r2": val_r2,
                "scaler_X_mean": scaler_X_mean,
                "scaler_X_std": scaler_X_std,
                "scaler_y_mean": scaler_y_mean,
                "scaler_y_std": scaler_y_std,
                # Full history up to this epoch so plot_fold_summary and
                # _load_fold_history in test_one_fold.py can read it back.
                "train_loss": history["train_loss"][:],
                "val_loss": history["val_loss"][:],
                "train_r2": history["train_r2"][:],
                "val_r2_history": history["val_r2"][:],
                "train_mae": history["train_mae"][:],
                "val_mae": history["val_mae"][:],
            }, best_model_path)
        else:
            patience_counter += 1
        
        # Early stopping: only start counting patience after both LR warmup
        # and gate warmup are complete — firing before that is premature since
        # the model hasn't yet reached its full training regime.
        warmup_done = epoch > max(warmup_epochs, GATE_WARMUP_EPOCHS)
        if early_stopping and warmup_done and patience_counter >= early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            print(f"    No improvement for {early_stopping_patience} epochs")
            print(f"    Best Val R²: {best_val_r2:+.4f}")
            break
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train R²={train_r2:+.4f} MAE={train_mae:.4f} | "
                  f"Val R²={val_r2:+.4f} MAE={val_mae:.4f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
    
    print(f"\nBest Val R²: {best_val_r2:+.4f}")
    return str(best_model_path), history


def diff_model_multi_fold_cv_train_test_optimized(
    distance_matrix_gics: np.ndarray,
    sector_ids: torch.Tensor,
    model_cfg: dict,
    n_splits: int = 9,
    max_train_size: int = 504,
    test_size: int = 126,
    gradient_accumulation_steps: int = 2,
    use_amp: bool = False,
    # Training hyperparameters forwarded to train_with_validation_optimized
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gate_lr_mult: float = 0.05,
    early_stopping: bool = False,
    early_stopping_patience: int = 15,
    max_epochs: int = None,
    gate_warmup_epochs: int = 10,
    warmup_epochs: int = 5,
):
    """
    Memory-optimized multi-fold cross-validation.

    Args:
        gradient_accumulation_steps: Simulate larger batches (e.g., 2 → effective batch_size = 32)
        use_amp: Use mixed precision training (FP16) to reduce memory
        lr: Learning rate passed to each fold's optimizer.
        weight_decay: L2 penalty passed to each fold's optimizer.
        gate_lr_mult: Multiplier on gate learning rate.
        early_stopping: Whether to enable early stopping per fold.
        early_stopping_patience: Epochs without improvement before stopping.
        max_epochs: Maximum training epochs per fold (overrides p.num_epochs when set).
        gate_warmup_epochs: Number of epochs gates are frozen at start.
        warmup_epochs: Number of epochs for linear LR warmup at the start of training.
    """
    # Prepare data
    X = distance_matrix_gics[:-1, np.newaxis, :, :]
    y = (distance_matrix_gics[1:] - distance_matrix_gics[:-1])[:, np.newaxis, :, :]
    
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size)
    
    fold_models = []
    all_fold_history = []
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X_tensor), start=1):
        print(f"\n{'='*70}")
        print(f"Fold {fold}/{n_splits}")
        print(f"{'='*70}")
        
        X_train_raw, X_val_raw = X_tensor[train_index], X_tensor[val_index]
        y_train_raw, y_val_raw = y_tensor[train_index], y_tensor[val_index]
        
        # Fold-wise standardization
        X_mean = X_train_raw.mean().item()
        X_std = X_train_raw.std(unbiased=False).item()
        y_mean = y_train_raw.mean().item()
        y_std = y_train_raw.std(unbiased=False).item()
        
        if X_std == 0.0 or y_std == 0.0:
            raise ValueError(f"Fold {fold}: standard deviation is zero")
        
        X_train = (X_train_raw - X_mean) / X_std
        y_train = (y_train_raw - y_mean) / y_std
        X_val = (X_val_raw - X_mean) / X_std
        y_val = (y_val_raw - y_mean) / y_std
        
        # DataLoaders (no pin_memory for CPU to save memory)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        pin_memory = torch.cuda.is_available() and p.USE_GPU
        
        train_loader = DataLoader(
            train_dataset, batch_size=p.BATCH_SIZE, shuffle=True,
            num_workers=p.NUM_WORKERS, pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=p.BATCH_SIZE, shuffle=False,
            num_workers=p.NUM_WORKERS, pin_memory=pin_memory,
        )
        
        # Create model
        model = SmallDataDecoderViT(
            **model_cfg,
            sector_ids=sector_ids,
        )
        
        # Train
        epochs_to_run = max_epochs if max_epochs is not None else p.num_epochs
        model_path, fold_history = train_with_validation_optimized(
            model, train_loader, val_loader, fold,
            epochs=epochs_to_run,
            lr=lr,
            weight_decay=weight_decay,
            gate_lr_mult=gate_lr_mult,
            warmup_epochs=warmup_epochs,
            scaler_X_mean=X_mean, scaler_X_std=X_std,
            scaler_y_mean=y_mean, scaler_y_std=y_std,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            gate_warmup_epochs=gate_warmup_epochs,
        )
        
        fold_models.append(model_path)
        all_fold_history.append(fold_history)
        
        # Cleanup
        del model, train_loader, val_loader, train_dataset, val_dataset
        del X_train, y_train, X_val, y_val
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Return best model overall
    best_fold_idx = np.argmax([
        max(h["val_r2"]) for h in all_fold_history
    ])
    
    print(f"\n{'='*70}")
    print(f"Best fold: {best_fold_idx + 1}")
    print(f"{'='*70}\n")
    
    return fold_models[best_fold_idx], all_fold_history
