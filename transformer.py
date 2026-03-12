"""
Decoder-only Vision Transformer for Small Datasets (~2000 images)
==================================================================

Input:  (B, C, 457, 457)
Output: (B, C, 457, 457)

Key modifications for small-data regimes
-----------------------------------------
1. Standard Linear Patch Embedding
   Each patch (C × p × p pixels) is flattened, layer-normalised, and
   projected to embed_dim with a single linear layer.  The previously
   used Shifted Patch Tokenization (SPT) has been removed for the
   following reasons specific to GICS-reordered distance matrices:

     (a) Cross-sector contamination: 52.4% of all 841 patches in the
         29×29 grid have at least one SPT shifted crop that crosses a
         GICS sector boundary, importing stocks from unrelated industries.
     (b) 160× noise amplification: the variance of a shift that crosses
         a sector boundary is 160× higher than a within-sector shift,
         meaning SPT injects far more noise than signal at boundaries.
     (c) Small-sector problem: 6 of 11 GICS sectors have fewer than
         36 stocks.  The fixed shift of p/2 = 8 stocks covers 22–44% of
         these sectors, almost guaranteeing boundary crossing.
     (d) Redundancy with LSA: the LSA locality bias (learnable per-head
         weight on a normalised Gaussian distance matrix) already provides
         the local-neighbourhood context that SPT was designed to supply,
         without the boundary-noise side-effect.

   SPT code is preserved below (class ShiftedPatchTokenization) for
   reference and to allow easy A/B comparison if desired.

2. Locality Self-Attention (LSA) with learnable temperature
   Each attention head has a learnable per-head temperature scalar.  A
   normalised Gaussian distance bias is added to attention logits so
   nearby patches (same GICS sector after reordering) are preferred
   early in training.  No causal mask is applied — the 29×29 patch grid
   represents spatial positions within one distance matrix snapshot (one
   trading day), not a temporal sequence.

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
* SPT & LSA : Lee et al., "Vision Transformer for Small-Size Datasets", 2021
              https://arxiv.org/abs/2112.13492
* LayerScale : Touvron et al., "Going Deeper with Image Transformers", 2021
* Stoch Depth: Huang et al., "Deep Networks with Stochastic Depth", 2016
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


def _gaussian_distance_bias(grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
    """
    Returns a (N, N) matrix of *normalised* negative squared L2 distances
    between patch centre coordinates.

    Values lie in [-1, 0]:
      -  0 on the diagonal (same patch, distance = 0)
      - -1 for the maximally distant patch pair (corner to corner)

    Normalising to [-1, 0] makes locality_weight a true interpolation knob:
      locality_weight = 0  →  no spatial bias (pure content attention)
      locality_weight = k  →  distant patches suppressed by up to k logit units

    Without normalisation the raw squared distances scale as O(G²) (up to ~1568
    for a 29×29 grid), which caused the bias to dominate the attention logits
    and led to severe overfocusing (ratio > 5× in all blocks).
    """
    gy, gx = torch.meshgrid(
        torch.linspace(0, 1, grid_h, device=device),
        torch.linspace(0, 1, grid_w, device=device),
        indexing="ij",
    )
    coords = torch.stack([gy.flatten(), gx.flatten()], dim=-1)  # (N, 2)
    diff   = coords.unsqueeze(0) - coords.unsqueeze(1)           # (N, N, 2)
    dist2  = (diff ** 2).sum(-1)                                  # (N, N)
    return -dist2 / dist2.max()                                   # (N, N) in [-1, 0]


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

    Replaces ShiftedPatchTokenization (SPT) for GICS-reordered distance
    matrices.  See module docstring for the full rationale.

    The LSA locality bias already provides the local-neighbourhood
    context that SPT was designed to supply, without the cross-sector
    noise that SPT's fixed-stride shifts introduce at GICS boundaries.

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
        padded_size: int,   # kept for API compatibility with ShiftedPatchTokenization
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.padded_size = padded_size
        patch_dim = in_channels * patch_size * patch_size  # e.g. 1×16×16 = 256

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
        patches = self._patchify(x)          # (B, N, C×p×p)
        return self.proj(self.norm(patches)) # (B, N, embed_dim)


# ============================================================
# Shifted Patch Tokenization (SPT) — preserved for reference
# ============================================================

class ShiftedPatchTokenization(nn.Module):
    """
    PRESERVED FOR REFERENCE — not used in the current model.

    Each patch token is formed from the concatenation of:
        - the patch itself
        - the same patch shifted left, right, up, down (by half a patch size)
    This gives each token a 5× richer feature set covering its immediate
    neighbourhood.

    Why it was removed
    ------------------
    SPT was designed for natural images where adjacent pixels belong to
    the same physical surface (spatial continuity holds).  For
    GICS-reordered distance matrices the analogous assumption — that
    adjacent stocks are related — holds only *within* a sector block.
    Across sector boundaries (which occur at 52.4% of patches) the
    shifted crops mix unrelated industries, injecting noise with 160×
    higher variance than within-sector shifts.  The LSA locality bias
    already captures intra-sector local structure without this side-effect.

    To re-enable SPT: swap StandardPatchEmbed for ShiftedPatchTokenization
    in SmallDataDecoderViT.__init__ and update the patch_dim accordingly.
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

        # 5 shifted crops × (C × p × p) → embed_dim
        patch_dim = 5 * in_channels * patch_size * patch_size

        self.norm = nn.LayerNorm(patch_dim)
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def _shift(self, x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Shift the image by (dy, dx) pixels using reflect padding."""
        if dy > 0:
            x = F.pad(x, (0, 0, dy, 0))[:, :, :x.shape[2], :]
        elif dy < 0:
            x = F.pad(x, (0, 0, 0, -dy))[:, :, -x.shape[2]:, :]
        if dx > 0:
            x = F.pad(x, (dx, 0, 0, 0))[:, :, :, :x.shape[3]]
        elif dx < 0:
            x = F.pad(x, (0, -dx, 0, 0))[:, :, :, -x.shape[3]:]
        return x

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        gh, gw = H // p, W // p
        x = x.reshape(B, C, gh, p, gw, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, gh * gw, C * p * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p2 = self.patch_size // 2
        shifts = [
            x,
            self._shift(x,  p2,  0),
            self._shift(x, -p2,  0),
            self._shift(x,  0,  p2),
            self._shift(x,  0, -p2),
        ]
        patches = torch.cat([self._patchify(s) for s in shifts], dim=-1)
        return self.proj(self.norm(patches))


# ============================================================
# Locality Self-Attention (LSA)
# ============================================================

class LocalitySelfAttention(nn.Module):
    """
    Multi-head self-attention with:
      - Learnable per-head temperature (replaces fixed sqrt(d_k) scaling)
      - Learnable per-head locality bias weight (one scalar per head,
        normalised to [-1, 0])

    No causal mask is applied here. The 841 tokens (29×29) represent
    spatial patch positions within a *single* distance matrix snapshot
    (one trading day). There is no temporal ordering within a frame, so
    every patch should be free to attend to every other patch.

    Using a per-head locality_weight lets each head independently learn
    how much spatial proximity matters — some heads may attend globally
    while others remain local, producing the head diversity that
    multihead attention is designed to exploit.

    locality_strength is initialised to 0.1 so the bias is weak at the
    start of training, giving the random QKV projections room to drive
    head divergence before spatial preferences are learned.
    """

    def __init__(
        self,
        embed_dim:   int,
        num_heads:   int,
        grid_h:      int,
        grid_w:      int,
        attn_drop:   float = 0.0,
        locality_strength: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.grid_h    = grid_h
        self.grid_w    = grid_w

        # Learnable per-head temperature (initialised to 1/sqrt(d_k))
        init_temp = math.log(self.head_dim ** -0.5)
        self.temperature = nn.Parameter(
            torch.full((num_heads, 1, 1), init_temp)
        )

        # Learnable per-head locality bias weight
        self.locality_weight = nn.Parameter(
            torch.full((num_heads,), locality_strength)
        )

        self.qkv       = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)

        # Cached locality bias (rebuilt if device changes)
        self._cached_bias: Optional[torch.Tensor] = None
        self._cached_device: Optional[torch.device] = None

    def _locality_bias(self, device: torch.device) -> torch.Tensor:
        if self._cached_bias is None or self._cached_device != device:
            self._cached_bias   = _gaussian_distance_bias(self.grid_h, self.grid_w, device)
            self._cached_device = device
        return self._cached_bias  # (N, N), values in [-1, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, N, D)

        scale = self.temperature.exp()                     # (H, 1, 1)
        attn  = (q @ k.transpose(-2, -1)) * scale         # (B, H, N, N)

        loc  = self._locality_bias(x.device)                        # (N, N)
        lw   = self.locality_weight.view(self.num_heads, 1, 1)      # (H, 1, 1)
        attn = attn + lw * loc                                      # (B, H, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
        embed_dim:    int,
        num_heads:    int,
        grid_h:       int,
        grid_w:       int,
        mlp_ratio:    float = 4.0,
        attn_drop:    float = 0.0,
        proj_drop:    float = 0.0,
        drop_path:    float = 0.0,
        ls_init:      float = 1e-2,
        locality_strength: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = LocalitySelfAttention(
            embed_dim, num_heads, grid_h, grid_w, attn_drop, locality_strength
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
    Decoder-only Vision Transformer tuned for small datasets (~2 000 samples).

    Patch tokenisation uses StandardPatchEmbed (single linear projection).
    ShiftedPatchTokenization (SPT) was removed because GICS-reordered
    distance matrices have unequal sector sizes — the fixed p/2 shift
    crosses sector boundaries in 52.4% of patches, injecting noise with
    160× higher variance than within-sector shifts.  The LSA locality
    bias already provides the local-neighbourhood context that SPT was
    designed to supply.  See module docstring for full details.

    Parameters
    ----------
    in_channels        : Input (and output) channel count.
    img_size           : Spatial size of the square input image.  Default 457.
    patch_size         : Patch size in [16, 32].  Default 16.
    embed_dim          : Transformer embedding dimension.
    depth              : Number of transformer decoder blocks.
    num_heads          : Number of attention heads.
    mlp_ratio          : MLP hidden-dim multiplier.
    attn_drop          : Attention dropout.
    proj_drop          : MLP / projection dropout.
    drop_path_rate     : Maximum stochastic-depth drop probability
                         (linearly increases across blocks).
    ls_init_value      : LayerScale initialisation value.  Default 1e-2.
    locality_strength  : Initial weight of the locality bias per head.
    """

    def __init__(
        self,
        in_channels:       int   = 1,
        img_size:          int   = 457,
        patch_size:        int   = 16,
        embed_dim:         int   = 192,
        depth:             int   = 6,
        num_heads:         int   = 3,
        mlp_ratio:         float = 4.0,
        attn_drop:         float = 0.0,
        proj_drop:         float = 0.1,
        drop_path_rate:    float = 0.05,
        ls_init_value:     float = 1e-2,
        locality_strength: float = 0.1,
    ):
        super().__init__()
        assert 16 <= patch_size <= 32, "patch_size must be in [16, 32]"

        self.in_channels  = in_channels
        self.img_size     = img_size
        self.patch_size   = patch_size
        self.padded_size  = _next_multiple(img_size, patch_size)
        # Input is always square (457×457) so grid_h == grid_w.
        self.grid_h = self.grid_w = self.padded_size // patch_size
        self.num_patches  = self.grid_h * self.grid_w

        # ── Standard patch embedding (replaced SPT) ───────────────────────
        # patch_dim = C × p × p = 1 × 16 × 16 = 256
        # Linear 256 → 192  (vs SPT's 1280 → 384 → 192)
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
                self.grid_h, self.grid_w,
                mlp_ratio, attn_drop, proj_drop,
                drop_path=dpr[i],
                ls_init=ls_init_value,
                locality_strength=locality_strength,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ── Pixel reconstruction head ──────────────────────────────────────
        # patch_dim = C × p × p = 256
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
        x  = x.permute(0, 3, 1, 4, 2, 5)          # (B, C, gh, p, gw, p)
        return x.reshape(B, C, gh * p, gw * p)

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        # 1. Pad + tokenize
        x_pad  = self._pad(x)                       # (B, C, P, P)
        tokens = self.patch_embed(x_pad)            # (B, N, embed_dim)
        tokens = tokens + self.pos_embed

        # 2. Transformer blocks (full bidirectional attention in each)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # 3. Reconstruct pixels
        pixels = self.head(tokens)                  # (B, N, C*p*p)
        out    = self._unpatchify(pixels)           # (B, C, P, P)

        # 4. Crop to original size
        return out[:, :, :H, :W]                    # (B, C, 457, 457)


# ============================================================
# Recommended configs
# ============================================================

def small_data_vit_tiny(in_channels: int = 1, **kwargs) -> SmallDataDecoderViT:
    """~5.5 M params — training config; single-channel inputs, ~2 000 samples."""
    return SmallDataDecoderViT(
        in_channels=in_channels,
        embed_dim=192, depth=6, num_heads=3,
        proj_drop=0.1, drop_path_rate=0.05,
        **kwargs,
    )


def small_data_vit_small(in_channels: int = 3, **kwargs) -> SmallDataDecoderViT:
    """~22 M params — larger variant for 3-channel inputs or bigger datasets."""
    return SmallDataDecoderViT(
        in_channels=in_channels,
        embed_dim=384, depth=8, num_heads=6,
        proj_drop=0.1, drop_path_rate=0.1,
        **kwargs,
    )


def small_data_vit_base(in_channels: int = 3, **kwargs) -> SmallDataDecoderViT:
    """~86 M params — use with strong augmentation / pre-training."""
    return SmallDataDecoderViT(
        in_channels=in_channels,
        embed_dim=768, depth=12, num_heads=12,
        proj_drop=0.2, drop_path_rate=0.2,
        **kwargs,
    )


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    configs = [
        ("tiny",  small_data_vit_tiny,  1),
        ("small", small_data_vit_small, 3),
    ]

    for patch_size in (16, 32):
        for name, factory, in_ch in configs:
            model = factory(in_channels=in_ch, patch_size=patch_size).to(device)
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
                f"embed={model.patch_embed.__class__.__name__} | "
                f"forward={elapsed*1000:.1f} ms | "
                f"output={tuple(y.shape)} ✓"
            )
        print()

    print("All checks passed.")

    print("""
Actual training config (main.py / training_and_validation_functions.py)
------------------------------------------------------------------------
model        : small_data_vit_tiny  (embed_dim=192, depth=6, num_heads=3)
patch_embed  : StandardPatchEmbed   (256 → 192, replaces SPT 1280 → 384 → 192)
in_channels  : 1  (single-channel z-scored GICS-reordered distance matrix)
img_size     : 457  (padded to 464 = 29×16 before tokenisation)
optimizer    : AdamW — 3 param groups:
                 decay    lr=1e-4, wd=1e-2  (weight matrices)
                 no-decay lr=1e-4, wd=0     (biases, LayerNorm)
                 gamma    lr=1e-3, wd=0     (LayerScale γ — 10× boost)
scheduler    : none (early stopping, patience=10)
epochs       : up to 100 per fold
batch size   : configured via parameters.BATCH_SIZE
cv           : TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)
loss         : MSE
ls_init      : 1e-2
locality     : per-head weight, init=0.1, bias normalised to [-1, 0]
GICS order   : stocks reordered by GICS sector before training
               (see extract_distance_matrices.reorder_by_gics)
""")
