"""
Decoder-only Vision Transformer for Small Datasets (~2000 images)
==================================================================

Input:  (B, C, 457, 457)
Output: (B, C, 457, 457)

Key modifications for small-data regimes
-----------------------------------------
1. Self-Patch Tokenization (SPT)
   Patches are enriched with shifted-patch features (left/right/up/down neighbors
   concatenated before projection) so each token carries local context, reducing
   the burden on attention to learn low-level structure from scratch.

2. Locality Self-Attention (LSA) with learnable temperature
   Each attention head has a learnable per-head temperature scalar.  A local
   Gaussian distance bias is added to attention logits so nearby patches are
   naturally preferred early in training, preventing attention collapse on
   small datasets.

   NOTE: No causal mask is applied within a single frame. The 29×29 patch grid
   represents spatial positions within one distance matrix (one trading day),
   not a temporal sequence — every patch may attend to every other patch freely.
   Temporal ordering is enforced at the data level (the model is trained on
   consecutive day pairs t → t+1), not inside the attention mechanism.

3. Stochastic Depth (DropPath)
   Per-layer stochastic depth acts as a powerful regularizer equivalent to an
   ensemble of shallower networks.

4. LayerScale
   Per-channel learnable scale on residual branches stabilises training on
   small data by initialising residual contributions near zero.

5. Explicit padding + crop
   457 → padded to nearest multiple of patch_size before patchification,
   cropped back after reconstruction.  Works for any patch_size in [16, 32].

References
----------
* SPT & LSA : Lee et al., "Vision Transformer for Small-Size Datasets", 2021
              https://arxiv.org/abs/2112.13492
* LayerScale : Touvron et al., "Going Deeper with Image Transformers", 2021
* Stoch Depth: Huang et al., "Deep Networks with Stochastic Depth", 2016
"""

import math
from functools import partial
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
# Self-Patch Tokenization (SPT)
# ============================================================

class ShiftedPatchTokenization(nn.Module):
    """
    Each patch token is formed from the concatenation of:
        - the patch itself
        - the same patch shifted left, right, up, down (by half a patch size)
    This gives each token a 5× richer feature set covering its immediate
    neighbourhood, which is critical when data is scarce.

    Projection is split into two sequential linear layers with a GELU to
    give non-linear mixing before the main transformer.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size:  int,
        embed_dim:   int,
        padded_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.padded_size = padded_size

        # 5 shifted crops × (C × p × p) → embed_dim
        patch_dim = 5 * in_channels * patch_size * patch_size

        self.norm  = nn.LayerNorm(patch_dim)
        self.proj  = nn.Sequential(
            nn.Linear(patch_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def _shift(self, x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Shift the image by (dy, dx) pixels using reflect padding."""
        p = self.patch_size
        # positive dy → shift down (pad top, crop bottom)
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
        x = x.permute(0, 2, 4, 1, 3, 5)   # (B, gh, gw, C, p, p)
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
        patches = torch.cat([self._patchify(s) for s in shifts], dim=-1)  # (B, N, 5*C*p*p)
        return self.proj(self.norm(patches))                                # (B, N, embed_dim)


# ============================================================
# Locality Self-Attention (LSA)
# ============================================================

class LocalitySelfAttention(nn.Module):
    """
    Multi-head self-attention with:
      - Learnable per-head temperature (replaces fixed sqrt(d_k) scaling)
      - Learnable per-head locality bias weight (one scalar per head, normalised to [-1, 0])

    No causal mask is applied here. The 841 tokens (29×29) represent spatial
    patch positions within a *single* distance matrix snapshot (one trading
    day). There is no temporal ordering within a frame, so every patch should
    be free to attend to every other patch. Temporal ordering is handled at
    the data level: the model receives day t as input and predicts day t+1.

    Using a per-head locality_weight (rather than a single shared scalar) lets
    each head independently learn how much spatial proximity matters. This
    breaks the symmetry that caused all heads to focus on the same region when
    the bias was shared — some heads may learn to attend globally while others
    remain local, producing the head diversity that multihead attention is
    designed to exploit.

    locality_strength is initialised to 0.1 (down from 1.0) so the bias is
    weak at the start of training, giving the random QKV projections room to
    drive head divergence before spatial preferences are learned.
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

        # Learnable per-head locality bias weight (one scalar per head).
        # Per-head weights let each head independently learn how much to favour
        # nearby patches, breaking the symmetry that caused all heads to focus
        # on the same spatial region when a single shared scalar was used.
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

        # Learnable temperature scaling
        scale = self.temperature.exp()                     # (H, 1, 1)
        attn  = (q @ k.transpose(-2, -1)) * scale         # (B, H, N, N)

        # Additive locality bias: nearby patches get a boost, far ones a penalty.
        # Bias is normalised to [-1, 0] so each head's locality_weight is interpretable
        # as "how many logit units to penalise the most distant patch".
        # locality_weight is (H,) → reshape to (H, 1, 1) for broadcasting over (B, H, N, N).
        loc  = self._locality_bias(x.device)                        # (N, N)
        lw   = self.locality_weight.view(self.num_heads, 1, 1)      # (H, 1, 1)
        attn = attn + lw * loc                                      # (B, H, N, N)

        # NOTE: No causal mask. Patches represent spatial positions within a
        # single frame, not a time series — full bidirectional attention is correct.

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
    Initialised near zero so early training is stable (Touvron et al., 2021).
    """

    def __init__(self, dim: int, init_value: float = 1e-4):
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
        ls_init:      float = 1e-4,
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
    Decoder-only Vision Transformer tuned for small datasets (~2 000 images).

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
    ls_init_value      : LayerScale initialisation value.
    locality_strength  : Initial weight of the locality bias.
    """

    def __init__(
        self,
        in_channels:       int   = 3,
        img_size:          int   = 457,
        patch_size:        int   = 16,
        embed_dim:         int   = 384,
        depth:             int   = 8,
        num_heads:         int   = 6,
        mlp_ratio:         float = 4.0,
        attn_drop:         float = 0.0,
        proj_drop:         float = 0.1,
        drop_path_rate:    float = 0.1,
        ls_init_value:     float = 1e-4,
        locality_strength: float = 0.1,
    ):
        super().__init__()
        assert 16 <= patch_size <= 32, "patch_size must be in [16, 32]"

        self.in_channels  = in_channels
        self.img_size     = img_size
        self.patch_size   = patch_size
        self.padded_size  = _next_multiple(img_size, patch_size)
        self.grid_h = self.grid_w = self.padded_size // patch_size
        self.num_patches  = self.grid_h * self.grid_w

        # ---- Shifted-patch tokenizer ----
        self.patch_embed = ShiftedPatchTokenization(
            in_channels, patch_size, embed_dim, self.padded_size
        )

        # ---- Learned positional embeddings ----
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ---- Stochastic depth schedule ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # ---- Transformer blocks ----
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

        # ---- Pixel reconstruction head ----
        patch_dim = in_channels * patch_size * patch_size
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        # 1. Pad + tokenize via SPT
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

def small_data_vit_tiny(in_channels: int = 3, **kwargs) -> SmallDataDecoderViT:
    """~6 M params — fastest to train, good baseline."""
    return SmallDataDecoderViT(
        in_channels=in_channels,
        embed_dim=192, depth=6, num_heads=3,
        proj_drop=0.1, drop_path_rate=0.05,
        **kwargs,
    )


def small_data_vit_small(in_channels: int = 3, **kwargs) -> SmallDataDecoderViT:
    """~22 M params — recommended for 2 000-image datasets."""
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

    for patch_size in (16, 32):
        model = small_data_vit_small(in_channels=3, patch_size=patch_size).to(device)
        x = torch.randn(2, 3, 457, 457, device=device)

        t0 = time.time()
        with torch.no_grad():
            y = model(x)
        elapsed = time.time() - t0

        assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"

        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"patch_size={patch_size:2d} | "
            f"padded={model.padded_size} | "
            f"grid={model.grid_h}×{model.grid_w} | "
            f"N={model.num_patches:4d} patches | "
            f"params={n_params:,} | "
            f"forward={elapsed*1000:.1f} ms | "
            f"output={tuple(y.shape)} ✓"
        )

    print("\nAll checks passed.")

    # -----------------------------------------------------------------
    # Training-recipe hint
    # -----------------------------------------------------------------
    print("""
Recommended training recipe for 2 000 images
---------------------------------------------
optimizer   : AdamW, lr=1e-3, weight_decay=0.05
scheduler   : cosine decay with 5-epoch warmup, 200 epochs total
augmentation: RandAugment(n=2, m=9) + Mixup(alpha=0.2) + CutMix(alpha=1.0)
batch size  : 32–64 (gradient accumulation if memory constrained)
loss        : task-specific (e.g. L1 / MSE for reconstruction, CE for labels)
extra       : EMA of weights (decay=0.999) for better generalisation
""")
