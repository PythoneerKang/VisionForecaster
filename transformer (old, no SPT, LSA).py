import parameters as p
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_multiple(n: int, divisor: int) -> int:
    return math.ceil(n / divisor) * divisor


def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Returns an additive causal mask of shape (seq_len, seq_len).
    Positions that should be masked out are set to -inf.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)   # upper triangle → -inf
    return mask                           # lower triangle (incl. diag) → 0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (autoregressive) mask."""

    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, N, D)

        # Causal mask (rebuilt each forward pass; cache if performance matters)
        causal = _make_causal_mask(N, x.device)           # (N, N)

        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B, H, N, N)
        attn = attn + causal                               # broadcast over B, H
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerDecoderBlock(nn.Module):
    """A single pre-norm decoder block (no cross-attention — decoder-only style)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadCausalSelfAttention(embed_dim, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = FeedForward(embed_dim, mlp_ratio, proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DecoderOnlyViT(nn.Module):
    """
    Decoder-only Vision Transformer with causal attention masking.

    Parameters
    ----------
    in_channels   : Number of input (and output) channels.
    img_size      : Spatial size of the (square) input image.  Default 457.
    patch_size    : Patch height/width.  Must be in [16, 32].  Default 16.
    embed_dim     : Transformer embedding dimension.
    depth         : Number of transformer blocks.
    num_heads     : Number of attention heads.
    mlp_ratio     : MLP hidden-dim multiplier.
    attn_drop     : Attention dropout probability.
    proj_drop     : MLP / projection dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 457,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert 16 <= patch_size <= 32, "patch_size must be in [16, 32]"

        self.in_channels = in_channels
        self.img_size    = img_size
        self.patch_size  = patch_size

        # Pad img_size to next multiple of patch_size
        self.padded_size = _next_multiple(img_size, patch_size)
        self.num_patches_per_axis = self.padded_size // patch_size
        self.num_patches = self.num_patches_per_axis ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding  (linear projection of flattened patches)
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Pixel reconstruction head
        self.head = nn.Linear(embed_dim, patch_dim)

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
        """Pad spatial dims to self.padded_size."""
        h, w = x.shape[-2], x.shape[-1]
        pad_h = self.padded_size - h
        pad_w = self.padded_size - w
        if pad_h > 0 or pad_w > 0:
            # Pad on the right / bottom (F.pad uses reversed dim order)
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H_pad, W_pad)  →  (B, N, patch_dim)"""
        B, C, H, W = x.shape
        p = self.patch_size
        # Reshape into patch grid
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)            # (B, gh, gw, C, p, p)
        x = x.reshape(B, self.num_patches, C * p * p)
        return x

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_dim)  →  (B, C, H_pad, W_pad)"""
        B = tokens.shape[0]
        C = self.in_channels
        p = self.patch_size
        g = self.num_patches_per_axis
        x = tokens.reshape(B, g, g, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)            # (B, C, g, p, g, p)
        x = x.reshape(B, C, self.padded_size, self.padded_size)
        return x

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, C, 457, 457)

        Returns
        -------
        out : (B, C, 457, 457)  — same shape as input
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, (
            f"Expected spatial size {self.img_size}x{self.img_size}, got {H}x{W}"
        )

        # 1. Pad → patchify → embed
        x_pad   = self._pad(x)                          # (B, C, P, P)
        patches = self._patchify(x_pad)                  # (B, N, patch_dim)
        tokens  = self.patch_embed(patches)              # (B, N, embed_dim)
        tokens  = tokens + self.pos_embed                # add positional encoding

        # 2. Causal transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # 3. Project back to pixel space & fold
        pixels  = self.head(tokens)                      # (B, N, patch_dim)
        out_pad = self._unpatchify(pixels)               # (B, C, P, P)

        # 4. Crop back to original spatial size
        out = out_pad[:, :, :H, :W]                      # (B, C, 457, 457)
        return out
