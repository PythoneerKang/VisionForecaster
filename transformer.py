import parameters as p
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SPT(nn.Module):
    """ Shifted Patch Tokenization for better spatial encoding on small datasets """
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        # We shift in 4 directions to capture overlapping spatial info
        self.proj = nn.Conv2d(in_chans * 5, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [B, C, H, W]
        p = self.patch_size
        # Shifts: Left-Up, Right-Up, Left-Down, Right-Down
        x_lu = F.pad(x, (p//2, 0, p//2, 0))[:, :, :-p//2, :-p//2]
        x_ru = F.pad(x, (0, p//2, p//2, 0))[:, :, :-p//2, p//2:]
        x_ld = F.pad(x, (p//2, 0, 0, p//2))[:, :, p//2:, :-p//2]
        x_rd = F.pad(x, (0, p//2, 0, p//2))[:, :, p//2:, p//2:]
        
        x = torch.cat([x, x_lu, x_ru, x_ld, x_rd], dim=1)
        return self.proj(x) # [B, embed_dim, H/p, W/p]

class LSA(nn.Module):
    """ Locality Self-Attention to improve convergence on small datasets """
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.full((heads, 1, 1), (dim // heads) ** -0.5))
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # LSA: Learnable temperature scaling + Masking diagonal (self-token) + Causal masking (for decoder-only model)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # # Masking the diagonal helps the model focus on other local patches
        # mask = torch.eye(N, device=x.device).bool()
        # dots.masked_fill_(mask, float('-inf'))

        # Get sequence length
        sz = dots.size(-1)
        
        # COMBINED MASK: diagonal=-1 masks the diagonal AND the upper triangle.
        # This enforces causality AND prevents self-attention (common in LSA).
        mask = torch.tril(torch.ones(sz, sz), diagonal=-1).to(q.device)
        
        # Apply mask: 0s in the mask (future/self) become -inf
        dots = dots.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(dots, dim=-1)
        attn = self.attn_drop(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LSA(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionForecaster(nn.Module):
    def __init__(self, img_size=457, patch_size=16, in_chans=1, embed_dim=256, depth=6, heads=8, mlp_dim=512):
        super().__init__()
        # Calculate padding to make 457 divisible by patch_size (e.g., 457 -> 464)
        self.pad = (patch_size - img_size % patch_size) % patch_size
        self.grid_size = (img_size + self.pad) // patch_size
        num_patches = self.grid_size ** 2

        self.spt = SPT(patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, heads, mlp_dim) for _ in range(depth)
        ])
        
        self.head = nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        self.patch_size = patch_size
        self.in_chans = in_chans

    def forward(self, x):
        # 1. Padding to handle 457x457
        x = F.pad(x, (0, self.pad, 0, self.pad))
        
        # 2. Tokenization (SPT)
        x = self.spt(x) # [B, embed_dim, GH, GW]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x += self.pos_embed
        
        # 3. Transformer Layers (with LSA)
        for layer in self.layers:
            x = layer(x)
            
        # 4. Reconstruction
        x = self.head(x) # [B, N, p*p*C]
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                      h=self.grid_size, w=self.grid_size, 
                      p1=self.patch_size, p2=self.patch_size, c=self.in_chans)
        
        # 5. Crop back to original size
        return x[:, :, :457, :457]