from parameters import *
from extract_distance_matrices import (
    extract_distance_matrix,
    reorder_by_gics,
    get_gics_sector_boundaries,
    build_patch_sector_ids,
)
from transformer import SmallDataDecoderViT
from training_and_validation_functions import diff_model_multi_fold_cv_train_test_optimized

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import torch
import gc


# ─────────────────────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CFG = dict(
    in_channels=1,

    # patch_size=16 gives 841 patches (29×29 grid) and a 256-dim patch space.
    # Combined with ConvPatchEmbed this costs only embed_dim*(256+3) parameters
    # for the embedding layer — far cheaper than the previous flat linear at
    # patch_size=24 (576-dim patches, 28k params in embedding alone).
    patch_size=16,
    embed_dim=64,               # increased from 32: more representational capacity
                                # now that the gradient path through the head is clean
    depth=1,
    num_heads=2,                # head_dim = 64/2 = 32

    # Moderate regularisation — previous run had too much (dropout 0.4,
    # weight_decay 0.1) which combined with near-zero LayerScale gradients
    # prevented any learning.
    proj_drop=0.1,
    attn_drop=0.1,
    drop_path_rate=0.05,

    ls_init_value=1e-2,         # back to 1e-2: gives stronger gradient signal
                                # through the residual branches from the start
    gate_init=0.0,              # sigmoid(0)=0.5: equal positional/content blend
                                # at init, letting both streams contribute early

    use_checkpoint=True,
    attention_chunk_size=128,
)

# ─────────────────────────────────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_CFG = dict(
    lr=1e-4,                    # higher than previous 3e-5: model was not
                                # learning at all — gradients ~1e-8
    weight_decay=1e-2,          # reduced from 0.1: previous value was so
                                # strong it suppressed all weight updates
    gate_lr_mult=0.1,           # gates learn slower than other params

    early_stopping=True,
    # Patience counts from epoch 1, but warmup_epochs=5 means the model
    # hasn't reached full LR yet. Setting patience=20 ensures the model
    # trains at full LR for at least 15 epochs before stopping.
    early_stopping_patience=20,

    max_epochs=100,             # restored: 50 was too short given slow learning

    gradient_accumulation_steps=4,
    use_amp=False,

    gate_warmup_epochs=5,       # freeze gates for first 5 epochs while the
                                # content stream stabilises
    warmup_epochs=5,
)


if __name__ == "__main__":
    import math
    # Set random seeds
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"Random seeds set to {RANDOM_SEED}")
    
    # CPU-specific optimizations
    if not USE_GPU:
        torch.set_num_threads(TORCH_NUM_THREADS)
        torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)
        print(f"CPU threads: {TORCH_NUM_THREADS} intra-op, {TORCH_NUM_INTEROP_THREADS} inter-op")
    
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"  patch_size:        {MODEL_CFG['patch_size']}  (ConvPatchEmbed — cheap embedding)")
    print(f"  embed_dim:         {MODEL_CFG['embed_dim']}")
    print(f"  depth:             {MODEL_CFG['depth']}")
    print(f"  num_heads:         {MODEL_CFG['num_heads']}  (head_dim={MODEL_CFG['embed_dim']//MODEL_CFG['num_heads']})")
    print(f"  ls_init_value:     {MODEL_CFG['ls_init_value']}")
    print(f"  gate_init:         {MODEL_CFG['gate_init']}  (sigmoid→{1/(1+math.exp(-MODEL_CFG['gate_init'])):.2f})")
    print(f"  proj_drop:         {MODEL_CFG['proj_drop']}")
    print(f"  attn_drop:         {MODEL_CFG['attn_drop']}")
    print(f"  drop_path_rate:    {MODEL_CFG['drop_path_rate']}")
    print(f"  lr:                {TRAINING_CFG['lr']}")
    print(f"  weight_decay:      {TRAINING_CFG['weight_decay']}")
    print(f"  early_stopping:    patience={TRAINING_CFG['early_stopping_patience']}")
    print(f"  max_epochs:        {TRAINING_CFG['max_epochs']}")
    print("="*70 + "\n")

    # ── Accurate parameter count ───────────────────────────────────────────
    patch_size  = MODEL_CFG["patch_size"]
    embed_dim   = MODEL_CFG["embed_dim"]
    num_heads   = MODEL_CFG["num_heads"]
    depth       = MODEL_CFG["depth"]
    in_channels = MODEL_CFG["in_channels"]

    padded      = math.ceil(457 / patch_size) * patch_size
    grid        = padded // patch_size
    num_patches = grid * grid
    patch_dim   = in_channels * patch_size * patch_size

    # ConvPatchEmbed: Conv2d weight + bias + LayerNorm(embed_dim)
    p_patch_embed = embed_dim * in_channels * patch_size**2 + embed_dim + embed_dim * 2

    p_pos_embed   = num_patches * embed_dim

    # Per DecoderBlock
    p_attn  = embed_dim * (embed_dim * 3) + embed_dim * 3   # QKV weight + bias
    p_attn += embed_dim * embed_dim + embed_dim              # out proj weight + bias
    p_attn += num_heads                                      # gate_logit
    p_ln1   = embed_dim * 2
    p_ffn   = embed_dim * (embed_dim * 4) + embed_dim * 4   # fc1 weight + bias
    p_ffn  += embed_dim * 4 * embed_dim + embed_dim          # fc2 weight + bias
    p_ln2   = embed_dim * 2
    p_ls1   = embed_dim
    p_ls2   = embed_dim
    p_block = p_attn + p_ln1 + p_ffn + p_ln2 + p_ls1 + p_ls2

    p_final_norm = embed_dim * 2

    # Decoder head: single Linear(embed_dim → patch_dim)
    p_head  = embed_dim * patch_dim + patch_dim

    total_params = p_patch_embed + p_pos_embed + p_block * depth + p_final_norm + p_head

    print(f"Model architecture:")
    print(f"  Patches:                     {num_patches} ({grid}×{grid} grid)")
    print(f"  Embed dim:                   {embed_dim}  |  head_dim: {embed_dim // num_heads}")
    print(f"  Depth:                       {depth} layer(s)")
    print(f"  Patch embedding (Conv):      {p_patch_embed:,} params")
    print(f"  Positional embedding:        {p_pos_embed:,} params")
    print(f"  Transformer block ×{depth}:       {p_block * depth:,} params")
    print(f"  Decoder head:                {p_head:,} params")
    print(f"  ── Total:                    {total_params:,} params")
    print(f"  Training samples/fold:       ~504")
    print(f"  Samples/parameter ratio:     {504/total_params:.4f}")
    print()
    
    # ── Step 1: Extract distance matrices ─────────────────────────────────
    distance_matrix = extract_distance_matrix()
    
    print("Distance matrix loaded:")
    print(f"  Shape: {distance_matrix.shape}")
    print(f"  NaN check: {np.isnan(distance_matrix).any()}")
    print()
    
    # ── Step 2: Reorder by GICS ───────────────────────────────────────────
    distance_matrix_gics, tickers_gics, sector_labels = reorder_by_gics(
        distance_matrix
    )
    
    sector_boundaries = get_gics_sector_boundaries(sector_labels)
    
    print("GICS sectors:")
    for name, start, end in sector_boundaries:
        print(f"  {name:<30}  [{start:>3}, {end:>3})  n={end - start}")
    print()
    
    del distance_matrix
    gc.collect()
    
    # ── Step 3: Build patch sector IDs ─────────────────────────────────────
    sector_ids = build_patch_sector_ids(sector_labels, patch_size=patch_size)
    
    print(f"Patch sector IDs: {sector_ids.shape}, {sector_ids.unique().numel()} groups")
    print()
    
    # ── Step 4: Train with multi-fold CV ───────────────────────────────────
    print("="*70)
    print("STARTING TRAINING WITH IMPROVED CONFIGURATION")
    print("="*70)
    print()
    
    model_path, all_fold_history = diff_model_multi_fold_cv_train_test_optimized(
        distance_matrix_gics,
        sector_ids,
        MODEL_CFG,
        **TRAINING_CFG,
    )
    
    # ── Step 5: Interpretability ───────────────────────────────────────────
    print("\n" + "="*70)
    print("GENERATING INTERPRETABILITY PLOTS")
    print("="*70 + "\n")
    
    from model_interpretability import ModelInterpreter, plot_fold_summary
    
    # 5a. Training summary
    plot_fold_summary(all_fold_history, save_path="fold_summary_improved.png")
    
    # 5b. Load best model
    best_model = SmallDataDecoderViT(
        **MODEL_CFG,
        sector_ids=sector_ids,
    )
    best_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    best_model.load_state_dict(
        best_ckpt["model_state_dict"] if isinstance(best_ckpt, dict) 
        and "model_state_dict" in best_ckpt else best_ckpt
    )
    
    interp = ModelInterpreter(best_model, save_dir=".")
    
    # 5c. Prepare validation sample
    scaler_X_mean = best_ckpt.get("scaler_X_mean") if isinstance(best_ckpt, dict) else None
    scaler_X_std = best_ckpt.get("scaler_X_std") if isinstance(best_ckpt, dict) else None
    scaler_y_mean = best_ckpt.get("scaler_y_mean") if isinstance(best_ckpt, dict) else None
    scaler_y_std = best_ckpt.get("scaler_y_std") if isinstance(best_ckpt, dict) else None
    
    if scaler_X_mean is not None and scaler_y_mean is not None:
        X_scaled = (distance_matrix_gics[:-1] - scaler_X_mean) / scaler_X_std
        y_scaled = ((distance_matrix_gics[1:] - distance_matrix_gics[:-1]) - scaler_y_mean) / scaler_y_std
    else:
        scaler_mean = best_ckpt.get("scaler_mean") if isinstance(best_ckpt, dict) else None
        scaler_std = best_ckpt.get("scaler_std") if isinstance(best_ckpt, dict) else None
        if scaler_mean is not None:
            X_scaled = (distance_matrix_gics[:-1] - scaler_mean) / scaler_std
            y_scaled = (distance_matrix_gics[1:] - scaler_mean) / scaler_std
        else:
            X_scaled = distance_matrix_gics[:-1]
            y_scaled = distance_matrix_gics[1:]
    
    X = X_scaled[:, np.newaxis, :]
    y = y_scaled[:, np.newaxis, :]
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    
    tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)
    *_, (_, last_val_idx) = tscv.split(X_t)
    
    X_val = X_t[last_val_idx]
    y_val = y_t[last_val_idx]
    
    sample_x = X_val[-1:]
    sample_y = y_val[-1:]
    
    # Null model baseline
    if scaler_y_mean is not None and scaler_y_std is not None and float(scaler_y_std) != 0.0:
        null_value = float(-scaler_y_mean) / float(scaler_y_std)
    else:
        null_value = 0.0
    
    baseline_pred = torch.full_like(sample_y, null_value)
    baseline_val_full = torch.full_like(y_val, null_value)
    
    # 5d. Generate interpretation plots
    print("Generating visualizations...")
    
    # Attention maps (only plot once for single-layer models)
    num_layers = len(best_model.blocks)
    interp.plot_attention_maps(sample_x, layer=0)
    
    if num_layers > 1:
        # Only plot last layer separately if model has multiple layers
        interp.plot_attention_maps(
            sample_x,
            layer=num_layers - 1,
            filename="attention_maps_last_block.png",
        )
        interp.plot_attention_maps_overlay(
            sample_x,
            layer=num_layers - 1,
            filename="attention_maps_overlay_last_block.png",
        )
    
    interp.plot_attention_maps_overlay(sample_x, layer=0)
    interp.plot_gate_values()
    interp.plot_mean_attention_distance(sample_x)
    interp.plot_layerscale_gammas()
    interp.plot_attention_weights(sample_x)
    
    interp.plot_prediction_error_map(
        sample_x, sample_y,
        tickers=tickers_gics,
        sector_boundaries=sector_boundaries,
        X_val=X_val, y_val=y_val,
        baseline_pred=baseline_pred,
        baseline_val_full=baseline_val_full,
        baseline_name="zero (ΔD=0)",
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best model: {model_path}")
    print(f"Plots saved in current directory")
