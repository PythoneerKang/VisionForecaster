# SmallDataDecoderViT

> Decoder-only Vision Transformer with Standard Patch Embed · Sector-GPSA · LayerScale · DropPath — tuned for ~2,000 samples

A compact ViT architecture designed for predicting distance matrices on small datasets. Given a z-scored distance matrix at time *t*, the model outputs a predicted distance matrix at *t+1*.

---

## Architecture Overview

```
Input (B × 1 × 457 × 457)
        │
        │  reflect-pad 457 → 464
        ▼
Standard Patch Embedding
        │
        │  element-wise add
        ▼
Positional Embedding  (1, 841, 192)
        │
        ▼
┌────────────────────────────────────┐
│       DecoderBlock  × 6            │
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
| `embed_dim`        | 192          |
| `depth`            | 6            |
| `num_heads`        | 3            |
| `head_dim`         | 64           |
| `mlp_ratio`        | 4×           |
| `proj_drop`        | 0.1          |
| `drop_path_rate`   | 0.05         |
| `ls_init`          | 1e-2         |
| `gate_init`        | 2.0          |

---

## Key Components

### Standard Patch Embedding
Each patch is flattened, layer-normalised, and projected to `embed_dim` with a single linear layer:

```
(B, 841, 1 × 16 × 16) = (B, 841, 256)
    → LayerNorm
    → Linear 256 → 192
```

### Sector-Gated Positional Self-Attention (Sector-GPSA)
Each head interpolates between a **sectoral positional prior** and standard **content attention** via a learned gate:

```
output_h = g_h · (A_pos @ V)  +  (1 − g_h) · (A_content @ V)
```

where:
- **A_pos** — row-normalised sector-membership matrix. Each query patch attends uniformly over all patches in the same GICS sector. This is the positional prior: a direct encoding of the block-diagonal structure of the distance matrix.
- **A_content** — standard scaled-dot-product attention: softmax(Q·Kᵀ / √d).
- **g_h = sigmoid(λ_h)** — a learnable gate scalar per head. Initialised at λ=+2 so g≈0.88 (nearly fully positional at the start of training), providing stable low-variance gradients on the small dataset. Heads can learn g→0 (pure content) or remain near g→1 (pure positional/sector prior) depending on what the data supports.

#### Why sector membership rather than Euclidean distance (LSA)
Locality Self-Attention and vanilla ConViT-GPSA both use a Gaussian over grid distance as their positional prior. For a GICS-reordered distance matrix the "distance" between two patches is the number of stocks separating them in an alphabetical-within-sector ordering — a noisy Euclidean proxy for the true structure, which is *categorical*: same-sector pairs are strongly correlated, cross-sector pairs much less so, with hard discontinuities at every sector boundary (not a smooth gradient). Sector-GPSA directly encodes this domain knowledge.

No causal mask is applied. The 841 tokens represent spatial patch positions within a single distance matrix snapshot (one trading day), not a temporal sequence — every patch attends freely to every other patch.

```
QKV proj:  192 → 3 × 192  (no bias)
heads = 3,  head_dim = 64
out proj:  192 → 192
gate:      sigmoid(λ_h)  per head, shape (H,), init λ=+2
```

### LayerScale
A per-channel learnable scalar γ (shape 192, init `1e-2`) is applied to each residual branch output before the residual add. This stabilises gradient flow during early training on small datasets.

> **Note on init value:** The original paper uses `1e-4`, which is conservative for very deep networks (12+ blocks). With only 6 blocks, `1e-2` is safe and gives the optimizer a much stronger gradient signal, preventing gammas from staying frozen during training. A dedicated 10× higher learning rate is also applied to the gamma parameters via a separate AdamW parameter group.

### DropPath (Stochastic Depth)
Drop probability increases linearly across the 6 blocks from 0 → 0.05. At training time the network behaves as an ensemble of shallower subnetworks, reducing over-fitting.

### Feed-Forward Network (FFN)
Standard MLP with expansion ratio 4×:
```
Linear 192 → 768 → GELU → Dropout(0.1) → Linear 768 → 192 → Dropout(0.1)
```

### Pixel Reconstruction Head
Projects each of the 841 tokens back to its 16×16 patch of pixels:
```
Linear 192 → 192 → GELU → Linear 192 → 256  (256 = 1 × 16 × 16)
unpatchify → (B, 1, 464, 464) → crop → (B, 1, 457, 457)
```

---

## Data Flow

| Stage | Tensor Shape |
|---|---|
| Raw input | `(B, 1, 457, 457)` |
| After reflect-pad | `(B, 1, 464, 464)` |
| After patch embed | `(B, 841, 192)` |
| After pos. embed | `(B, 841, 192)` |
| After 6 DecoderBlocks | `(B, 841, 192)` |
| After final LayerNorm | `(B, 841, 192)` |
| After pixel head | `(B, 841, 256)` |
| After unpatchify + crop | `(B, 1, 457, 457)` |

**Training objective:** MSE loss between the predicted distance matrix and the ground-truth matrix at *t+1*.

---

## Padding Note

457 is not divisible by the patch size (16). The input is reflect-padded to 464 = 29 × 16 before tokenisation, then the reconstructed output is cropped back to 457 × 457.

---

## Interpretability

The following plots are generated by `model_interpretability.py` after training:

| Plot | Description |
|---|---|
| `fold_summary.png` | Train/val MSE and R² curves across all 9 folds |
| `attention_maps.png` | Per-head effective attention maps (first block) |
| `attention_maps_last_block.png` | Per-head effective attention maps (last block) |
| `attention_maps_overlay.png` | Colour-blended multi-head overlay (first block) |
| `attention_maps_overlay_last_block.png` | Colour-blended multi-head overlay (last block) |
| `gate_values.png` | Learned gate g=sigmoid(λ) heatmap per (block, head) |
| `mean_attention_distance.png` | Mean spatial distance of effective attention vs baselines |
| `layerscale_gammas.png` | LayerScale γ per block — residual branch health |
| `attention_weights.png` | Content attention weight distributions per block |
| `prediction_error_map.png` | Input / prediction / ground truth / error with GICS annotations |

---

## Inspiration

> Stéphane d'Ascoli, Hugo Touvron, Matthew Lerer, Armand Joulin, Piotr Bojanowski, Julien Garrigue. **ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases.** ICML 2021.
> arXiv: [2103.10697](https://arxiv.org/abs/2103.10697)

The GPSA gating mechanism is adapted from ConViT. The positional prior is replaced with a sector-membership matrix derived from GICS sector assignments, making the inductive bias domain-specific to GICS-reordered financial distance matrices rather than relying on Euclidean grid distance.

> Seung Hoon Lee, Seunghyun Lee, Byung Cheol Song. **Vision Transformer for Small-Size Datasets.** IEEE Access, 2022.
> DOI: [10.1109/ACCESS.2022.3220167](https://ieeexplore.ieee.org/document/9957006)

The small-data motivation (small dataset regime, LayerScale, DropPath) follows this work.
