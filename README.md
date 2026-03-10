# SmallDataDecoderViT

> Decoder-only Vision Transformer with SPT · LSA · LayerScale · DropPath — tuned for ~2,000 samples

A compact ViT architecture designed for predicting distance matrices on small datasets. Given a z-scored distance matrix at time *t*, the model outputs a predicted distance matrix at *t+1*.

---

## Architecture Overview

```
Input (B × 1 × 457 × 457)
        │
        │  reflect-pad 457 → 464
        ▼
Shifted Patch Tokenization (SPT)
        │
        │  element-wise add
        ▼
Positional Embedding  (1, 841, 192)
        │
        ▼
┌───────────────────────────────┐
│      DecoderBlock  × 6        │
│  ┌───────────┐ ┌───────────┐  │
│  │    LSA    │ │   FFN     │  │
│  │+LayerScale│ │+LayerScale│  │
│  │+DropPath  │ │+DropPath  │  │
│  └───────────┘ └───────────┘  │
└───────────────────────────────┘
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
| `ls_init`          | 1e-4         |

---

## Key Components

### Shifted Patch Tokenization (SPT)
Each patch token is enriched with 4 diagonally-shifted neighbours before projection, providing local context without relying solely on attention. Five crops (original + 4 shifts of p/2) are concatenated:

```
(B, 841, 5 × 1 × 16 × 16) = (B, 841, 1280)
    → LayerNorm
    → Linear 1280 → 384
    → GELU
    → Linear 384 → 192
```

### Locality Self-Attention (LSA)
Full causal self-attention with two small-data-friendly modifications:
- **Learnable per-head temperature** scalar, rather than fixed `1/√d`.
- **Gaussian distance bias** — `locality_weight × −‖Δcoord‖²` softly encourages attention to nearby patches.
- **Causal (lower-triangular) mask** enforces autoregressive ordering: patch *i* may only attend to patches 0…*i*.

```
QKV proj:  192 → 3 × 192  (no bias)
heads = 3,  head_dim = 64
out proj:  192 → 192
```

### LayerScale
A per-channel learnable scalar γ (shape 192, init `1e-4`) is applied to each residual branch output before the residual add. This stabilises gradient flow during early training on small datasets.

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
| After SPT | `(B, 841, 192)` |
| After pos. embed | `(B, 841, 192)` |
| After 6 DecoderBlocks | `(B, 841, 192)` |
| After final LayerNorm | `(B, 841, 192)` |
| After pixel head | `(B, 841, 256)` |
| After unpatchify + crop | `(B, 1, 457, 457)` |

**Training objective:** MSE loss between the predicted distance matrix and the ground-truth matrix at *t+1*.

---

## Padding Note

457 is not divisible by the patch size (16). The input is reflect-padded to 464 = 29 × 16 before tokenisation, then the reconstructed output is cropped back to 457 × 457.
