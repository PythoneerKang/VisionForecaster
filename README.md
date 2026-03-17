# SmallDataDecoderViT

> Decoder-only Vision Transformer with Conv Patch Embed · Sector-GPSA · LayerScale · DropPath — tuned for ~500 samples

A compact ViT architecture designed for predicting changes in distance matrices on small datasets. Given a fold-standardised distance matrix at time *t*, the model outputs the **predicted change** ΔD̂_t = D_{t+1} − D_t (not the next level D_{t+1}).

---

## Architecture Overview

```
Input (B × 1 × 457 × 457)
        │
        │  reflect-pad 457 → 464
        ▼
Convolutional Patch Embedding
        │
        │  element-wise add
        ▼
Positional Embedding  (1, 841, 64)
        │
        ▼
┌────────────────────────────────────┐
│       DecoderBlock  × 1            │
│  ┌──────────────┐ ┌─────────────┐  │
│  │  SectorGPSA  │ │     FFN     │  │
│  │ +LayerScale  │ │ +LayerScale │  │
│  │ +DropPath    │ │ +DropPath   │  │
│  └──────────────┘ └─────────────┘  │
└────────────────────────────────────┘
        │
        ▼
Final LayerNorm
        │
        ▼
Pixel Reconstruction Head
        │
        │  unpatchify + crop 464 → 457
        ▼
Output (B × 1 × 457 × 457)
```

For the full interactive diagram, open [`docs/index.html`](https://pythoneerkang.github.io/VisionForecaster/) in a browser.

---

## Model Parameters

| Hyperparameter     | Value        |
|--------------------|--------------|
| `in_channels`      | 1            |
| `img_size`         | 457          |
| `padded_size`      | 464          |
| `patch_size`       | 16           |
| `grid`             | 29 × 29      |
| `N patches`        | 841          |
| `embed_dim`        | 64           |
| `depth`            | 1            |
| `num_heads`        | 2            |
| `head_dim`         | 32           |
| `mlp_ratio`        | 4×           |
| `attn_drop`        | 0.1          |
| `proj_drop`        | 0.1          |
| `drop_path_rate`   | 0.05         |
| `ls_init`          | 1e-2         |
| `gate_init`        | 0.0          |

---

## Key Components

### Convolutional Patch Embedding

Uses a `Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)` followed by a post-projection `LayerNorm(embed_dim)`. The norm operates in `embed_dim` space (cheap, fixed cost of `2 × embed_dim` parameters) rather than raw patch space, making it far cheaper than a flat linear embedding for any patch size.

```
Conv2d(1, 64, kernel_size=16, stride=16)   →   64 × 256 + 64 = 16,448 weights
LayerNorm(64)                               →   128 weights
Total patch embedding:                          16,576 params
```

### Sector-Gated Positional Self-Attention (Sector-GPSA)

Each head interpolates between a **sectoral positional prior** and standard **content attention** via a learned gate:

```
output_h = g_h · (A_pos @ V)  +  (1 − g_h) · (A_content @ V)
```

where:
- **A_pos** — row-normalised sector-pair membership matrix. Each patch at grid position (r, c) is assigned a group based on `frozenset({majority_sector(row_stocks), majority_sector(col_stocks)})`, so patch (r,c) and patch (c,r) always share the same group, preserving distance matrix symmetry.
- **A_content** — standard scaled-dot-product attention: softmax(Q·Kᵀ / √d).
- **g_h = sigmoid(λ_h)** — a learnable gate scalar per head. Initialised at λ=0 so g=0.5 (equal positional/content blend at the start of training).

```
QKV proj:  64 → 3 × 64  (bias=True)
heads = 2,  head_dim = 32
out proj:  64 → 64
gate:      sigmoid(λ_h)  per head, shape (H,), init λ=0 → g=0.5
```

### LayerScale

A per-channel learnable scalar γ (shape 64, init `1e-2`) applied to each residual branch output before the residual add.

### DropPath (Stochastic Depth)

Fixed rate of 0.05. Drops the entire block at training time with probability 0.05.

### Feed-Forward Network (FFN)

Standard MLP with expansion ratio 4×:
```
Linear 64 → 256 → GELU → Dropout(0.1) → Linear 256 → 64 → Dropout(0.1)
```

### Pixel Reconstruction Head

A single linear projection directly from `embed_dim` to raw patch pixel space:
```
Linear(64 → 256)   (256 = 1 × 16 × 16)
unpatchify → (B, 1, 464, 464) → crop → (B, 1, 457, 457)
```

A two-layer head (`Linear→GELU→Linear`) was previously used but caused gradient death: the 8× expansion from `embed_dim=64` to `patch_dim=256` in the final layer diluted gradients across 256 output dimensions from only 64 inputs, producing near-zero gradients throughout the transformer. A single linear gives a direct gradient path from the loss to all earlier layers.

---

## Parameter Budget

| Component              | Parameters |
|------------------------|------------|
| Conv Patch Embedding   | 16,576     |
| Positional Embedding   | 53,824     |
| Transformer Block ×1   | 50,114     |
| Final LayerNorm        | 128        |
| Decoder Head           | 16,640     |
| **Total**              | **137,282** |

Training samples per fold: ~504 → **samples/parameter ratio: 0.0037**

---

## Data Flow

| Stage | Tensor Shape |
|---|---|
| Raw input | `(B, 1, 457, 457)` |
| After reflect-pad | `(B, 1, 464, 464)` |
| After conv patch embed | `(B, 841, 64)` |
| After pos. embed | `(B, 841, 64)` |
| After 1 DecoderBlock | `(B, 841, 64)` |
| After final LayerNorm | `(B, 841, 64)` |
| After pixel head | `(B, 841, 256)` |
| After unpatchify + crop | `(B, 1, 457, 457)` |

**Training objective:** MSE loss between the predicted change ΔD̂_t and the ground-truth change ΔD_t = D_{t+1} − D_t. The null model predicts ΔD = 0 (no change); performance is reported as R² relative to this null.

---

## Padding Note

457 is not divisible by the patch size (16). The input is reflect-padded to 464 = 29 × 16 before tokenisation, then the reconstructed output is cropped back to 457 × 457.

---

## Interpretability

The following plots are generated by `model_interpretability.py` after training:

| Plot | Description |
|---|---|
| `fold_summary_improved.png` | Train/val loss and R² curves across all 9 folds |
| `attention_maps.png` | Per-head effective attention maps (block 1) |
| `attention_maps_overlay.png` | Colour-blended multi-head overlay (block 1) |
| `gate_values.png` | Learned gate g=sigmoid(λ) heatmap per (block, head) |
| `mean_attention_distance.png` | Mean spatial distance of effective attention vs baselines |
| `bar_mean_attention_distance.png` | Bar chart of mean attention distance per block vs baselines |
| `layerscale_gammas.png` | LayerScale γ per block — residual branch health |
| `attention_weights.png` | Content attention weight distributions per block |
| `prediction_error_map.png` | Input D_t / predicted ΔD̂ / ground truth ΔD / signed error / per-pixel skill score with GICS annotations |

> Note: `attention_maps_last_block.png` and `attention_maps_overlay_last_block.png` are only generated when `depth > 1`.

---

## Inspiration

> Stéphane d'Ascoli, Hugo Touvron, Matthew Lerer, Armand Joulin, Piotr Bojanowski, Julien Garrigue. **ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases.** ICML 2021.
> arXiv: [2103.10697](https://arxiv.org/abs/2103.10697)

The GPSA gating mechanism is adapted from ConViT. The positional prior is replaced with a sector-membership matrix derived from GICS sector assignments, making the inductive bias domain-specific to GICS-reordered financial distance matrices rather than relying on Euclidean grid distance.

> Seung Hoon Lee, Seunghyun Lee, Byung Cheol Song. **Vision Transformer for Small-Size Datasets.** IEEE Access, 2022.
> DOI: [10.1109/ACCESS.2022.3220167](https://ieeexplore.ieee.org/document/9957006)

The small-data motivation (small dataset regime, LayerScale, DropPath) follows this work.
