# SmallDataDecoderViT

> Decoder-only Vision Transformer with Standard Patch Embed · LSA · LayerScale · DropPath — tuned for ~2,000 samples

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
| `ls_init`          | 1e-2         |
| `locality_strength`| 0.1          |

---

## Key Components

### Standard Patch Embedding
Each patch is flattened, layer-normalised, and projected to `embed_dim` with a single linear layer:

```
(B, 841, 1 × 16 × 16) = (B, 841, 256)
    → LayerNorm
    → Linear 256 → 192
```

### Locality Self-Attention (LSA)
Full bidirectional self-attention with two small-data-friendly modifications:
- **Learnable per-head temperature** scalar, rather than fixed `1/√d`.
- **Learnable per-head locality bias weight** — each head has its own `locality_weight` scalar (shape `(H,)`, init `0.1`), applied as `locality_weight_h × −‖Δcoord‖² / max(‖Δcoord‖²)`. The bias is normalised to `[−1, 0]`, so each head's weight is a direct logit-units knob: its value equals the suppression applied to the most distant patch for that head. Using a per-head weight lets heads independently learn how much spatial proximity matters, producing the head diversity that multihead attention is designed to exploit. The low init (`0.1`) keeps the bias weak at the start of training, giving the random QKV projections room to drive head divergence before spatial preferences are learned.

No causal mask is applied. The 841 tokens represent spatial patch positions within a single distance matrix snapshot (one trading day), not a temporal sequence — every patch attends freely to every other patch. Temporal ordering is enforced at the data level (input = day *t*, target = day *t+1*).

```
QKV proj:  192 → 3 × 192  (no bias)
heads = 3,  head_dim = 64
out proj:  192 → 192
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

## Inspiration

> Seung Hoon Lee, Seunghyun Lee, Byung Cheol Song. **Vision Transformer for Small-Size Datasets.** IEEE Access, 2022.
> DOI: [10.1109/ACCESS.2022.3220167](https://ieeexplore.ieee.org/document/9957006)

The LSA technique used in this architecture is adapted from this work, which introduced locality inductive bias mechanisms enabling ViTs to train effectively on small datasets without large-scale pre-training.
