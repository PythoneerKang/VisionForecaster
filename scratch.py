from modulefinder import test
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        # SPT concatenates 5 versions (Original + 4 shifts)
        self.proj = nn.Linear(in_channels * 5 * patch_size**2, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # Shift directions: (dx, dy)
        shifts = [ (0,0), (-p//2, -p//2), (p//2, -p//2), (-p//2, p//2), (p//2, p//2) ]
        shifted_images = []
        for dx, dy in shifts:
            shifted_images.append(torch.roll(x, shifts=(dy, dx), dims=(2, 3)))
        
        x = torch.cat(shifted_images, dim=1) # (B, C*5, H, W)
        x = x.unfold(2, p, p).unfold(3, p, p) # Patchify
        x = x.permute(0, 2, 3, 1, 4, 5).flatten(3) # (B, h, w, C*5*p*p)
        return self.proj(x.flatten(1, 2)) # (B, N, dim)

class LocalitySelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(heads, 1, 1)) # Learnable temperature
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.heads, -1).transpose(1, 2) for t in (q, k, v)]

        # Scaled Dot-Product with learnable temperature
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 1. Causal Mask
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        dots.masked_fill_(mask, float('-inf'))
        
        # 2. Locality: Mask diagonal to remove self-tokens
        diag_mask = torch.eye(N, device=x.device).bool()
        dots.masked_fill_(diag_mask, float('-inf'))

        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)

class DecoderViT(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, img_size=457, dim=64, depth=3):
        super().__init__()
        # Handle non-divisible 457 by padding to next multiple of patch_size
        self.pad_size = (patch_size - img_size % patch_size) % patch_size
        self.spt = ShiftedPatchTokenization(in_channels, dim, patch_size)
        
        num_patches = ((img_size + self.pad_size) // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        self.layers = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(dim), LocalitySelfAttention(dim), 
                           nn.LayerNorm(dim), nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))])
            for _ in range(depth)
        ])
        self.to_pixels = nn.Linear(dim, in_channels * patch_size**2)

    def forward(self, x):
        # Pad -> SPT -> Transformer -> Un-patch -> Crop
        x = F.pad(x, (0, self.pad_size, 0, self.pad_size))
        B, C, H, W = x.shape
        p = self.spt.patch_size
        
        x = self.spt(x) + self.pos_embedding
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
            
        x = self.to_pixels(x).view(B, H//p, W//p, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return x[:, :, :457, :457] # Restore original size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecoderViT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

def train_step(img_in, img_tgt):
    model.train()
    optimizer.zero_grad()
    output = model(img_in.to(device))
    loss = criterion(output, img_tgt.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def test_step(img_in, img_tgt):
    model.eval()
    with torch.no_grad():
        output = model(img_in.to(device))
        loss = criterion(output, img_tgt.to(device))
    return loss.item()

#test_loss = test_step()