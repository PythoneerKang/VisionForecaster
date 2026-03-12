"""
Decoder-only Vision Transformer for Small Datasets (~2000 images)
==================================================================

Input:  (B, C, 457, 457)
Output: (B, C, 457, 457)

Key modifications for small-data regimes
-----------------------------------------
1. Standard Linear Patch Embedding
   Each patch (C × p × p pixels) is flattened, layer-normalised, and
   projected to embed_dim with a single linear layer.

   ShiftedPatchTokenization (SPT) was removed because GICS-reordered
   distance matrices have unequal sector sizes — the fixed p/2 shift
   crosses sector boundaries in 52.4% of patches, injecting noise with
   160× higher variance than within-sector shifts.

2. Sector-Gated Positional Self-Attention (GPSA)
   Replaces Locality Self-Attention (LSA).  Each head interpolates
   between two attention distributions via a learnable gate g ∈ (0,1):

       output_h = g_h · (A_pos @ V)  +  (1 − g_h) · (A_content @ V)

   where:
     A_pos     = row-normalised sector-membership matrix — each query
                 attends uniformly over all patches in the same GICS
                 sector as itself.  This is the positional prior.
     A_content = standard scaled-dot-product attention (QKᵀ / √d).
     g_h       = sigmoid(λ_h), one learnable scalar per head.
                 Initialised to λ=+2 so heads start ~88% positional,
                 giving stable low-variance gradients early in training
                 on the small (~2 000-sample) dataset.

   Why sector membership rather than Euclidean distance (LSA / vanilla GPSA)
   --------------------------------------------------------------------------
   LSA and ConViT-GPSA both use a Gaussian over grid distance as their
   positional prior.  For a GICS-reordered distance matrix the "distance"
   between two patches is the number of stocks separating them in an
   alphabetical-within-sector ordering — a noisy Euclidean proxy for the
   true structure, which is *categorical*: same-sector pairs are strongly
   correlated, cross-sector pairs much less so, with hard discontinuities
   at every sector boundary (not a smooth gradient).

   Sector-GPSA directly encodes this domain knowledge:
     - The positional component is exactly the block-diagonal prior.
     - The gate lets each head smoothly escape the prior as data
       supports it, instead of forcing content attention from epoch 1.
     - Heads that learn g→1 act like intra-sector averagers;
       heads that learn g→0 act like standard content-attention heads.

3. Stochastic Depth (DropPath)
   Per-layer stochastic depth acts as a powerful regularizer equivalent
   to an ensemble of shallower networks.

4. LayerScale
   Per-channel learnable scale on residual branches stabilises training
   on small data.  Default init 1e-2 (not the paper's 1e-4, which is
   tuned for 12+ block networks; 6 blocks are safe at 1e-2).

5. Explicit padding + crop
   457 → padded to nearest multiple of patch_size before patchification,
   cropped back after reconstruction.

References
----------
* GPSA / ConViT : d'Ascoli et al., "ConViT: Improving Vision Transformers
                  with Soft Convolutional Inductive Biases", ICML 2021
                  https://arxiv.org/abs/2103.10697
* SPT & LSA     : Lee et al., "Vision Transformer for Small-Size Datasets",
                  IEEE Access 2022.  https://arxiv.org/abs/2112.13492
* LayerScale    : Touvron et al., "Going Deeper with Image Transformers", 2021
* Stoch Depth   : Huang et al., "Deep Networks with Stochastic Depth", 2016
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def _next_multiple(n: int, d: int) -> int:
    return math.ceil(n / d) * d


def _build_sector_positional_attn(
    sector_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the (N, N) positional attention matrix A_pos from sector IDs.

    A_pos[i, j] = 1 / |sector(i)|  if sector(i) == sector(j), else 0.

    Each row sums to 1 (valid probability distribution), so the positional
    component (A_pos @ V) is a uniform average over same-sector patches.

    Parameters
    ----------
    sector_ids : (N,) integer tensor — GICS sector index per patch.
    device     : target device.

    Returns
    -------
    (N, N) float32 tensor on `device`.
    """
    sector_ids = sector_ids.to(device)
    same = (sector_ids.unsqueeze(0) == sector_ids.unsqueeze(1)).float()  # (N, N)
    # Row-normalise: each row sums to 1 (uniform within sector)
    row_sum = same.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return same / row_sum  # (N, N)


# ============================================================
# Drop Path (Stochastic Depth)
# ============================================================

class DropPath(nn.Module):
    """Stochastic depth regularisation (Huang et al., 2016)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = torch.floor(noise + keep)
        return x * noise / keep


# ============================================================
# Standard Patch Embedding
# ============================================================

class StandardPatchEmbed(nn.Module):
    """
    Flatten each (C × p × p) patch → LayerNorm → single Linear → embed_dim.

    Architecture
    ------------
    (B, N, C×p×p)  →  LayerNorm  →  Linear(C×p×p → embed_dim)
                                  →  (B, N, embed_dim)
    """

    def __init__(
        self,
        in_channels: int,
        patch_size:  int,
        embed_dim:   int,
        padded_size: int,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.padded_size = padded_size
        patch_dim = in_channels * patch_size * patch_size

        self.norm = nn.LayerNorm(patch_dim)
        self.proj = nn.Linear(patch_dim, embed_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, C×p×p)  where N = (H/p) × (W/p)"""
        B, C, H, W = x.shape
        p = self.patch_size
        gh, gw = H // p, W // p
        x = x.reshape(B, C, gh, p, gw, p)
        x = x.permute(0, 2, 4, 1, 3, 5)   # (B, gh, gw, C, p, p)
        return x.reshape(B, gh * gw, C * p * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(x)
        return self.proj(self.norm(patches))


# ============================================================
# Sector-Gated Positional Self-Attention (Sector-GPSA)
# ============================================================

class SectorGPSA(nn.Module):
    """
    Sector-Gated Positional Self-Attention (Sector-GPSA).

    Each head h computes:

        A_pos_h     = sector_positional_attention(sector_ids)   # (N, N)
        A_content_h = softmax(Q_h · K_hᵀ / √d)                 # (N, N)
        g_h         = sigmoid(λ_h)                              # scalar ∈ (0,1)

        out_h = g_h · (A_pos_h @ V_h)  +  (1 − g_h) · (A_content_h @ V_h)

    A_pos is shared across heads (it depends only on sector_ids which are
    fixed), but each head has its own gate λ_h so heads can independently
    decide how much to rely on the sectoral prior vs. content.

    Parameters
    ----------
    embed_dim         : Model embedding dimension.
    num_heads         : Number of attention heads.
    sector_ids        : (N,) integer tensor — GICS sector index per patch.
                        Must be registered as a buffer so it moves with .to(device).
    attn_drop         : Dropout on content attention weights.
    gate_init         : Initial value of the raw gate logit λ (before sigmoid).
                        sigmoid(+2) ≈ 0.88, so heads start ~88% positional.
    """

    def __init__(
        self,
        embed_dim:   int,
        num_heads:   int,
        sector_ids:  torch.Tensor,   # (N,) int
        attn_drop:   float = 0.0,
        gate_init:   float = 2.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Learnable gate per head: g_h = sigmoid(lambda_h)
        # Initialised so heads start nearly fully positional
        self.gate_logit = nn.Parameter(
            torch.full((num_heads,), gate_init)
        )

        self.qkv       = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)

        # sector_ids: non-trainable buffer, moves with .to(device) automatically
        self.register_buffer("sector_ids", sector_ids.long())

        # _a_pos: pre-built (N, N) sector-membership attention matrix.
        # Stored as a non-persistent buffer so it:
        #   - moves with .to(device) automatically (no device-check overhead)
        #   - is excluded from state_dict (rebuilt from sector_ids on load)
        #   - is computed exactly once at construction, not every forward pass
        self.register_buffer(
            "_a_pos",
            _build_sector_positional_attn(sector_ids, sector_ids.device),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each (B, H, N, D)

        # ── Content attention ──────────────────────────────────────────────
        attn_content = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn_content = attn_content.softmax(dim=-1)
        attn_content = self.attn_drop(attn_content)
        v_content    = attn_content @ v                          # (B, H, N, D)

        # ── Positional attention ───────────────────────────────────────────
        # _a_pos is a pre-built (N, N) buffer — no device check, no rebuild.
        # einsum avoids unsqueeze+broadcast: PyTorch/BLAS sees a single
        # (N, N) × (B*H, N, D) call and picks the optimal BLAS path,
        # avoiding the implicit tensor replication that @ with (1,1,N,N)
        # would cause on CPU.
        v_pos = torch.einsum("mn,bhnd->bhmd", self._a_pos, v)   # (B, H, N, D)

        # ── Gate interpolation ─────────────────────────────────────────────
        # g shape: (H,) → (1, H, 1, 1) for broadcasting
        g   = self.gate_logit.sigmoid().view(1, self.num_heads, 1, 1)
        out = g * v_pos + (1.0 - g) * v_content                 # (B, H, N, D)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ============================================================
# Feed-Forward Network
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        h = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, h),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(h, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# LayerScale
# ============================================================

class LayerScale(nn.Module):
    """
    Learnable per-channel scale on residual branches.

    init_value is set to 1e-2 by default (changed from the paper's 1e-4).
    The original 1e-4 is tuned for very deep networks (12+ blocks); with
    only 6 blocks, 1e-2 is safe and gives the optimizer a stronger gradient
    signal so gammas don't stay frozen.
    """

    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# ============================================================
# Transformer Block
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim:   int,
        num_heads:   int,
        sector_ids:  torch.Tensor,
        mlp_ratio:   float = 4.0,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
        drop_path:   float = 0.0,
        ls_init:     float = 1e-2,
        gate_init:   float = 2.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = SectorGPSA(
            embed_dim, num_heads, sector_ids, attn_drop, gate_init
        )
        self.ls1   = LayerScale(embed_dim, ls_init)
        self.dp1   = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = FeedForward(embed_dim, mlp_ratio, proj_drop)
        self.ls2   = LayerScale(embed_dim, ls_init)
        self.dp2   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.dp2(self.ls2(self.ff(self.norm2(x))))
        return x


# ============================================================
# Main Model
# ============================================================

class SmallDataDecoderViT(nn.Module):
    """
    Decoder-only Vision Transformer with Sector-GPSA, tuned for small
    datasets (~2 000 samples) on GICS-reordered distance matrices.

    Attention mechanism
    -------------------
    Sector-Gated Positional Self-Attention (Sector-GPSA) replaces
    Locality Self-Attention (LSA).  Each head gates between:
      - A positional component: uniform attention within the same GICS sector
      - A content component:    standard scaled-dot-product attention
    Gate g_h = sigmoid(λ_h) is learned per head; λ initialised to +2
    so heads start ~88% positional, providing stable early-training gradients.

    Parameters
    ----------
    in_channels     : Input (and output) channel count.
    img_size        : Spatial size of the square input image.  Default 457.
    patch_size      : Patch size in [16, 32].  Default 16.
    embed_dim       : Transformer embedding dimension.
    depth           : Number of transformer decoder blocks.
    num_heads       : Number of attention heads.
    mlp_ratio       : MLP hidden-dim multiplier.
    attn_drop       : Attention dropout.
    proj_drop       : MLP / projection dropout.
    drop_path_rate  : Maximum stochastic-depth drop probability
                      (linearly increases across blocks).
    ls_init_value   : LayerScale initialisation value.  Default 1e-2.
    gate_init       : Initial gate logit λ.  sigmoid(2) ≈ 0.88.
    sector_ids      : (N,) integer tensor mapping each patch to its GICS
                      sector index.  Must be provided.
    """

    def __init__(
        self,
        in_channels:    int           = 1,
        img_size:       int           = 457,
        patch_size:     int           = 16,
        embed_dim:      int           = 192,
        depth:          int           = 6,
        num_heads:      int           = 3,
        mlp_ratio:      float         = 4.0,
        attn_drop:      float         = 0.0,
        proj_drop:      float         = 0.1,
        drop_path_rate: float         = 0.05,
        ls_init_value:  float         = 1e-2,
        gate_init:      float         = 2.0,
        sector_ids:     torch.Tensor  = None,
    ):
        super().__init__()
        assert 16 <= patch_size <= 32, "patch_size must be in [16, 32]"
        assert sector_ids is not None, (
            "sector_ids (N,) must be provided.  "
            "Call build_patch_sector_ids() from extract_distance_matrices.py."
        )

        self.in_channels  = in_channels
        self.img_size     = img_size
        self.patch_size   = patch_size
        self.padded_size  = _next_multiple(img_size, patch_size)
        self.grid_h = self.grid_w = self.padded_size // patch_size
        self.num_patches  = self.grid_h * self.grid_w

        # Validate sector_ids length
        assert len(sector_ids) == self.num_patches, (
            f"sector_ids length {len(sector_ids)} != num_patches {self.num_patches}"
        )

        # ── Patch embedding ───────────────────────────────────────────────
        self.patch_embed = StandardPatchEmbed(
            in_channels, patch_size, embed_dim, self.padded_size
        )

        # ── Learned positional embeddings ─────────────────────────────────
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Stochastic depth schedule ──────────────────────────────────────
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # ── Transformer blocks ─────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim, num_heads,
                sector_ids,
                mlp_ratio, attn_drop, proj_drop,
                drop_path=dpr[i],
                ls_init=ls_init_value,
                gate_init=gate_init,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ── Pixel reconstruction head ──────────────────────────────────────
        patch_dim = in_channels * patch_size * patch_size
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_dim),
        )

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────────
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        ph = self.padded_size - h
        pw = self.padded_size - w
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, C*p*p) → (B, C, padded_size, padded_size)"""
        B  = tokens.shape[0]
        C  = self.in_channels
        p  = self.patch_size
        gh = self.grid_h
        gw = self.grid_w
        x  = tokens.reshape(B, gh, gw, C, p, p)
        x  = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, gh * p, gw * p)

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        x_pad  = self._pad(x)
        tokens = self.patch_embed(x_pad)
        tokens = tokens + self.pos_embed

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        pixels = self.head(tokens)
        out    = self._unpatchify(pixels)
        return out[:, :, :H, :W]


# ============================================================
# Recommended configs
# ============================================================

def small_data_vit_tiny(
    in_channels: int = 1,
    sector_ids: torch.Tensor = None,
    **kwargs,
) -> SmallDataDecoderViT:
    """~5.5 M params — training config; single-channel inputs, ~2 000 samples."""
    return SmallDataDecoderViT(
        in_channels=in_channels,
        embed_dim=192, depth=6, num_heads=3,
        proj_drop=0.1, drop_path_rate=0.05,
        sector_ids=sector_ids,
        **kwargs,
    )


def small_data_vit_small(
    in_channels: int = 3,
    sector_ids: torch.Tensor = None,
    **kwargs,
) -> SmallDataDecoderViT:
    """~22 M params — larger variant for 3-channel inputs or bigger datasets."""
    return SmallDataDecoderViT(
        in_channels=in_channels,
        embed_dim=384, depth=8, num_heads=6,
        proj_drop=0.1, drop_path_rate=0.1,
        sector_ids=sector_ids,
        **kwargs,
    )


def small_data_vit_base(
    in_channels: int = 3,
    sector_ids: torch.Tensor = None,
    **kwargs,
) -> SmallDataDecoderViT:
    """~86 M params — use with strong augmentation / pre-training."""
    return SmallDataDecoderViT(
        in_channels=in_channels,
        embed_dim=768, depth=12, num_heads=12,
        proj_drop=0.2, drop_path_rate=0.2,
        sector_ids=sector_ids,
        **kwargs,
    )


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Minimal mock sector_ids for smoke test: 841 patches split evenly into
    # 11 GICS sectors (the real ids come from build_patch_sector_ids()).
    N_test = 29 * 29  # 841
    mock_sector_ids = torch.zeros(N_test, dtype=torch.long)
    boundaries = [0, 76, 152, 228, 304, 380, 456, 532, 608, 684, 760, 841]
    for idx, (s, e) in enumerate(zip(boundaries, boundaries[1:])):
        mock_sector_ids[s:e] = idx

    configs = [
        ("tiny",  small_data_vit_tiny,  1),
        ("small", small_data_vit_small, 3),
    ]

    for patch_size in (16, 32):
        for name, factory, in_ch in configs:
            # Recompute N for this patch_size
            padded = math.ceil(457 / patch_size) * patch_size
            N_ps   = (padded // patch_size) ** 2
            sids   = torch.zeros(N_ps, dtype=torch.long)
            chunk  = N_ps // 11
            for i in range(11):
                sids[i*chunk:min((i+1)*chunk, N_ps)] = i

            model = factory(
                in_channels=in_ch,
                patch_size=patch_size,
                sector_ids=sids,
            ).to(device)

            x = torch.randn(2, in_ch, 457, 457, device=device)
            t0 = time.time()
            with torch.no_grad():
                y = model(x)
            elapsed = time.time() - t0

            assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"

            n_params = sum(p.numel() for p in model.parameters())
            print(
                f"[{name:5s}] patch_size={patch_size:2d} | "
                f"padded={model.padded_size} | "
                f"grid={model.grid_h}×{model.grid_w} | "
                f"N={model.num_patches:4d} patches | "
                f"params={n_params:,} | "
                f"attn=SectorGPSA | "
                f"forward={elapsed*1000:.1f} ms | "
                f"output={tuple(y.shape)} ✓"
            )
        print()

    print("All checks passed.")

    print("""
Actual training config (main.py / training_and_validation_functions.py)
------------------------------------------------------------------------
model        : small_data_vit_tiny  (embed_dim=192, depth=6, num_heads=3)
attention    : SectorGPSA  (sector-gated positional self-attention)
               g_h = sigmoid(λ_h), init λ=+2 → g≈0.88 (nearly fully positional)
               positional prior: uniform attention within GICS sector
               content:          scaled-dot-product QKᵀ/√d
patch_embed  : StandardPatchEmbed  (256 → 192)
in_channels  : 1  (single-channel z-scored GICS-reordered distance matrix)
img_size     : 457  (padded to 464 = 29×16 before tokenisation)
optimizer    : AdamW — 3 param groups:
                 decay    lr=1e-4, wd=1e-2  (weight matrices)
                 no-decay lr=1e-4, wd=0     (biases, LayerNorm)
                 gamma    lr=1e-3, wd=0     (LayerScale γ — 10× boost)
               Note: gate_logit params fall in no-decay group (1-D)
scheduler    : none (early stopping, patience=10)
epochs       : up to 100 per fold
batch size   : configured via parameters.BATCH_SIZE
cv           : TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)
loss         : MSE
ls_init      : 1e-2
gate_init    : +2.0  (sigmoid ≈ 0.88 positional at init)
GICS order   : stocks reordered by GICS sector before training
               (see extract_distance_matrices.reorder_by_gics)
sector_ids   : patch→sector mapping from build_patch_sector_ids()
               in extract_distance_matrices.py
""")
