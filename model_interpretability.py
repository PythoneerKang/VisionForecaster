"""
model_interpretability.py
=========================
Extract and visualise interpretability signals from SmallDataDecoderViT.

Usage
-----
    from model_interpretability import ModelInterpreter

    # After training, load a saved fold model
    model = SmallDataDecoderViT(in_channels=1, embed_dim=192, depth=6, num_heads=3,
                                 proj_drop=0.1, drop_path_rate=0.05)
    model.load_state_dict(torch.load("model_fold_1.pth", map_location="cpu"))

    interp = ModelInterpreter(model)

    # --- Attention maps (needs a sample input) ---
    sample = next(iter(val_loader))[0][:1]          # shape (1, 1, 457, 457)
    interp.plot_attention_maps(sample, layer=0)      # all heads, first block
    interp.plot_mean_attention_distance(sample)      # how local each head is

    # --- Weight-level diagnostics (no input needed) ---
    interp.plot_layerscale_gammas()                  # residual branch health
    interp.plot_attention_temperatures()             # per-head sharpness
    interp.plot_locality_weights()                   # learned spatial bias strength
    interp.plot_locality_bias_scale(sample)          # overfocusing diagnostic

    # --- Training history ---
    from model_interpretability import plot_fold_summary
    plot_fold_summary(all_fold_history)              # MSE + R² across all folds
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Hook helper
# ──────────────────────────────────────────────────────────────────────────────

class _AttentionHook:
    """Registers a forward hook on a LocalitySelfAttention layer to capture
    the full attention weight tensor (B, H, N, N)."""

    def __init__(self):
        self.weights: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, layer):
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inputs, output):
        # Re-run the attention *without* dropout so weights are deterministic
        x = inputs[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scale = module.temperature.exp()
        attn = (q @ k.transpose(-2, -1)) * scale
        loc = module._locality_bias(x.device)
        attn = attn + module.locality_weight * loc
        causal = torch.triu(torch.full((N, N), float("-inf"), device=x.device), diagonal=1)
        attn = attn + causal
        self.weights = attn.softmax(dim=-1).detach().cpu()  # (B, H, N, N)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()


# ──────────────────────────────────────────────────────────────────────────────
# Main interpreter class
# ──────────────────────────────────────────────────────────────────────────────

class ModelInterpreter:
    """
    Interpretability toolkit for SmallDataDecoderViT.

    Parameters
    ----------
    model : SmallDataDecoderViT
        A trained (or partially trained) model instance.
    save_dir : str
        Directory where figures are saved (default: current directory).
    """

    def __init__(self, model, save_dir: str = "."):
        self.model = model
        self.save_dir = save_dir
        model.eval()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _blocks(self):
        return list(self.model.blocks)

    def _grid(self):
        return self.model.grid_h, self.model.grid_w

    def _savefig(self, fig, name: str):
        path = f"{self.save_dir}/{name}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    def _get_attn_weights(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Return attention weights (1, H, N, N) for a single sample."""
        hook = _AttentionHook()
        hook.register(self._blocks()[layer].attn)
        with torch.no_grad():
            self.model(x[:1])
        hook.remove()
        return hook.weights  # (1, H, N, N)

    # ── 1. Attention maps ────────────────────────────────────────────────────

    def plot_attention_maps(
        self,
        x: torch.Tensor,
        layer: int = 0,
        query_patch: Optional[int] = None,
        filename: str = "attention_maps.png",
    ):
        """
        Plot per-head attention maps for a given layer.

        For each head, shows the attention weight distribution from the
        *query_patch* token to all other tokens, reshaped to the 2-D grid.
        If query_patch is None, the centre patch is used.

        Parameters
        ----------
        x            : Input tensor (B, C, H, W) — only first sample is used.
        layer        : Which transformer block (0-indexed).
        query_patch  : Source patch index.  Default = centre patch.
        filename     : Output filename.
        """
        gh, gw = self._grid()
        N = gh * gw
        if query_patch is None:
            query_patch = N // 2

        attn = self._get_attn_weights(x, layer)  # (1, H, N, N)
        H = attn.shape[1]

        ncols = min(H, 4)
        nrows = math.ceil(H / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()
        fig.suptitle(
            f"Attention maps — Block {layer+1}  |  Query patch {query_patch}  "
            f"({gh}×{gw} grid)",
            fontsize=12, fontweight="bold",
        )

        for h in range(H):
            ax = axes[h]
            weights = attn[0, h, query_patch, :].numpy()  # (N,)
            grid = weights.reshape(gh, gw)
            im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
            ax.set_title(f"Head {h+1}", fontsize=9)
            ax.axis("off")
            # Mark the query patch
            qy, qx = divmod(query_patch, gw)
            ax.scatter([qx], [qy], c="red", s=60, marker="x", zorder=5)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes[H:]:
            ax.axis("off")

        plt.tight_layout()
        self._savefig(fig, filename)

    # ── 2. Mean attention distance ───────────────────────────────────────────

    def plot_mean_attention_distance(
        self,
        x: torch.Tensor,
        filename: str = "mean_attention_distance.png",
    ):
        """
        For each (layer, head) compute the average distance (in patch hops)
        between query and key weighted by attention probability.  Low values
        = locally focused; high values = globally attending.

        Also computes two exact baselines from the distance matrix itself:
          - Uniform baseline: mean pairwise L2 distance under uniform attention
            (i.e. what you'd expect if the model attends randomly).
          - Causal-uniform baseline: same, but restricted to the lower triangle
            (i.e. uniform attention under the causal mask your model uses).
        These are printed to stdout and drawn as reference lines on a companion
        bar chart so you can judge whether each head is more local or more
        global than chance.

        Produces two figures:
          1. Heat-map of shape (depth, num_heads)  [filename]
          2. Bar chart with baseline overlaid       [filename prefixed 'bar_']
        """
        gh, gw = self._grid()
        N = gh * gw
        depth = len(self._blocks())

        # ── Pre-compute patch-centre coordinates and full distance matrix ──
        gy = torch.arange(gh).float()
        gx = torch.arange(gw).float()
        cy, cx = torch.meshgrid(gy, gx, indexing="ij")
        coords = torch.stack([cy.flatten(), cx.flatten()], dim=-1)  # (N, 2)
        dist_mat = torch.cdist(coords, coords, p=2).numpy()          # (N, N)

        # ── Exact baselines ────────────────────────────────────────────────
        # 1. Uniform baseline: every (query, key) pair equally weighted
        uniform_baseline = dist_mat.mean()

        # 2. Causal-uniform baseline: lower triangle only (key <= query),
        #    matching what the causal mask allows.  Each row q has (q+1) valid
        #    keys (0..q), so we weight accordingly.
        causal_sum = 0.0
        causal_count = 0
        for q in range(N):
            # keys 0..q are visible; key == q contributes distance 0
            causal_sum += dist_mat[q, : q + 1].sum()
            causal_count += q + 1
        causal_baseline = causal_sum / causal_count

        print(f"\n── Mean Attention Distance Baselines (grid={gh}×{gw}, N={N}) ──")
        print(f"  Max possible L2 distance : {dist_mat.max():.4f}  "
              f"(corner-to-corner = {gh-1}√2 ≈ {(gh-1)*2**0.5:.4f})")
        print(f"  Uniform-attention baseline: {uniform_baseline:.4f}  "
              f"(expected distance if model attends uniformly over all N patches)")
        print(f"  Causal-uniform  baseline  : {causal_baseline:.4f}  "
              f"(expected distance under uniform attention + causal mask)")
        print(f"  → Values below {causal_baseline:.1f} indicate the model is MORE local than chance.")
        print(f"  → Values above {causal_baseline:.1f} indicate the model is MORE global than chance.\n")

        # ── Compute per-(layer, head) mean attention distance ──────────────
        mean_dist = np.zeros((depth, self._blocks()[0].attn.num_heads))

        for layer in range(depth):
            attn = self._get_attn_weights(x, layer)  # (1, H, N, N)
            H = attn.shape[1]
            for h in range(H):
                w = attn[0, h].numpy()  # (N, N) — w[q, k]
                mean_dist[layer, h] = (w * dist_mat).sum(axis=-1).mean()

        # ── Figure 1: heat-map ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(6, H), depth + 1))
        im = ax.imshow(mean_dist, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=dist_mat.max())
        ax.set_xticks(range(H))
        ax.set_xticklabels([f"H{h+1}" for h in range(H)])
        ax.set_yticks(range(depth))
        ax.set_yticklabels([f"Block {l+1}" for l in range(depth)])
        ax.set_xlabel("Attention Head")
        ax.set_title(
            f"Mean Attention Distance (patch hops)  —  grid {gh}×{gw}\n"
            f"Causal-uniform baseline = {causal_baseline:.2f}  |  "
            f"Uniform baseline = {uniform_baseline:.2f}  |  "
            f"Max = {dist_mat.max():.2f}",
            fontweight="bold", fontsize=9,
        )
        fig.colorbar(im, ax=ax, label="mean hop distance")

        for l in range(depth):
            for h in range(H):
                val = mean_dist[l, h]
                # White text if cell is dark, black if light
                text_col = "white" if val > dist_mat.max() * 0.6 else "black"
                ax.text(h, l, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=text_col, fontweight="bold")

        plt.tight_layout()
        self._savefig(fig, filename)

        # ── Figure 2: bar chart (mean across heads, per layer) + baselines ─
        mean_per_layer = mean_dist.mean(axis=1)   # (depth,)
        bar_filename = "bar_" + filename

        fig2, ax2 = plt.subplots(figsize=(max(7, depth + 2), 4))
        x_pos = np.arange(1, depth + 1)
        bars = ax2.bar(x_pos, mean_per_layer, color="#5C8DD6", edgecolor="white",
                       linewidth=0.6, label="Model (mean over heads)")

        ax2.axhline(uniform_baseline, color="#E53935", linewidth=1.5,
                    linestyle="--", label=f"Uniform baseline ({uniform_baseline:.2f})")
        ax2.axhline(causal_baseline, color="#FB8C00", linewidth=1.5,
                    linestyle="-.", label=f"Causal-uniform baseline ({causal_baseline:.2f})")
        ax2.axhline(0, color="black", linewidth=0.6)

        for bar, val in zip(bars, mean_per_layer):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax2.set_xlabel("Block")
        ax2.set_ylabel("Mean attention distance (patch hops)")
        ax2.set_title(
            "Mean Attention Distance per Block vs Baselines\n"
            "Below causal-uniform → more local than chance  |  "
            "Above → more global than chance",
            fontweight="bold",
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"Block {l+1}" for l in range(depth)])
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, dist_mat.max() * 1.05)
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._savefig(fig2, bar_filename)

    # ── 3. LayerScale gammas ─────────────────────────────────────────────────

    def plot_layerscale_gammas(self, filename: str = "layerscale_gammas.png"):
        """
        Plot the LayerScale γ values for each block.

        γ ≈ 0  → residual branch suppressed (block barely contributes).
        γ ≈ 1  → full-strength residual.

        Two lines per block: attention branch (ls1) and FFN branch (ls2).
        """
        blocks = self._blocks()
        depth = len(blocks)

        ls1_means, ls1_stds = [], []
        ls2_means, ls2_stds = [], []

        for blk in blocks:
            g1 = blk.ls1.gamma.detach().cpu().numpy()
            g2 = blk.ls2.gamma.detach().cpu().numpy()
            ls1_means.append(g1.mean())
            ls1_stds.append(g1.std())
            ls2_means.append(g2.mean())
            ls2_stds.append(g2.std())

        x = np.arange(1, depth + 1)
        fig, ax = plt.subplots(figsize=(max(7, depth + 1), 4))
        ax.errorbar(x, ls1_means, yerr=ls1_stds, marker="o", label="Attention branch (ls1)",
                    color="#2196F3", capsize=4)
        ax.errorbar(x, ls2_means, yerr=ls2_stds, marker="s", label="FFN branch (ls2)",
                    color="#F44336", capsize=4, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Block")
        ax.set_ylabel("LayerScale γ (mean ± std)")
        ax.set_title("LayerScale Gammas per Block\n"
                     "γ ≈ 0 → branch suppressed  |  γ ≈ 1 → full strength",
                     fontweight="bold")
        ax.set_xticks(x)
        ax.legend()
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._savefig(fig, filename)

    # ── 4. Attention temperatures ────────────────────────────────────────────

    def plot_attention_temperatures(self, filename: str = "attention_temperatures.png"):
        """
        Plot the learned per-head temperature for every block.

        High temperature exp(t) → sharper / more peaked attention.
        Low  temperature exp(t) → softer / more uniform attention.

        Shown as a heat-map (block × head).
        """
        blocks = self._blocks()
        depth = len(blocks)
        H = blocks[0].attn.num_heads

        temps = np.zeros((depth, H))
        for l, blk in enumerate(blocks):
            t = blk.attn.temperature.exp().detach().cpu().numpy()  # (H, 1, 1)
            temps[l] = t[:, 0, 0]

        fig, ax = plt.subplots(figsize=(max(6, H + 1), depth + 1))
        im = ax.imshow(temps, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(H))
        ax.set_xticklabels([f"H{h+1}" for h in range(H)])
        ax.set_yticks(range(depth))
        ax.set_yticklabels([f"Block {l+1}" for l in range(depth)])
        ax.set_xlabel("Attention Head")
        ax.set_title("Learned Attention Temperature exp(t)\n"
                     "High = sharp/focused  |  Low = soft/diffuse",
                     fontweight="bold")
        fig.colorbar(im, ax=ax, label="exp(temperature)")

        for l in range(depth):
            for h in range(H):
                ax.text(h, l, f"{temps[l, h]:.3f}", ha="center", va="center",
                        fontsize=8, color="black")

        plt.tight_layout()
        self._savefig(fig, filename)

    # ── 5. Locality bias weights ─────────────────────────────────────────────

    def plot_locality_weights(self, filename: str = "locality_weights.png"):
        """
        Bar chart of the learned locality_weight scalar per block.

        Positive & large → model leans on spatial proximity (local).
        Near zero / negative → model ignores or suppresses locality bias.
        """
        blocks = self._blocks()
        weights = [blk.attn.locality_weight.item() for blk in blocks]
        x = np.arange(1, len(blocks) + 1)

        colors = ["#4CAF50" if w > 0 else "#F44336" for w in weights]
        fig, ax = plt.subplots(figsize=(max(7, len(blocks) + 1), 4))
        bars = ax.bar(x, weights, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Block")
        ax.set_ylabel("Locality weight")
        ax.set_title("Learned Locality Bias Weight per Block\n"
                     "Green (+) = prefers local attention  |  Red (−) = suppresses it",
                     fontweight="bold")
        ax.set_xticks(x)
        for bar, val in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.01 * np.sign(val), f"{val:.3f}",
                    ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._savefig(fig, filename)

    # ── 5b. Locality bias scale diagnostic ───────────────────────────────────

    def plot_locality_bias_scale(
        self,
        x: torch.Tensor,
        filename: str = "locality_bias_scale.png",
    ):
        """
        Diagnose whether the locality bias dominates or is overwhelmed by the
        raw attention logits, on a per-block basis.

        For each block this computes:
          - logit_std   : std of (Q·Kᵀ × temperature) before any bias is added
          - bias_max    : max absolute value of (locality_weight × bias matrix)
          - ratio       : bias_max / logit_std  ← the key number

        Ratio interpretation
        --------------------
        < 1×   : bias is weak — locality barely affects attention
        1–5×   : bias meaningfully shapes attention but doesn't dominate
        5–20×  : bias strongly suppresses distant patches
        > 20×  : severe overfocusing — distant patches are effectively dead keys

        Also prints a per-block summary table to stdout.

        If any block has ratio > 5, a warning is printed recommending that the
        locality bias be normalised in transformer.py (see note in output).
        """
        blocks = self._blocks()
        depth  = len(blocks)
        gh, gw = self._grid()
        N      = gh * gw

        # Storage
        logit_stds  = np.zeros(depth)
        logit_mins  = np.zeros(depth)
        logit_maxs  = np.zeros(depth)
        bias_maxs   = np.zeros(depth)
        bias_mins   = np.zeros(depth)
        ratios      = np.zeros(depth)
        lw_vals     = np.zeros(depth)

        handles = []

        def make_hook(idx):
            def hook(module, inputs, _output):
                inp = inputs[0]
                B, N_, C = inp.shape
                qkv = module.qkv(inp).reshape(B, N_, 3, module.num_heads, module.head_dim)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

                scale       = module.temperature.exp()                    # (H,1,1)
                raw_logits  = (q @ k.transpose(-2, -1)) * scale           # (B,H,N,N)
                loc         = module._locality_bias(inp.device)           # (N,N)
                bias        = (module.locality_weight * loc).detach().cpu().numpy()

                raw_np = raw_logits.detach().cpu().numpy()

                logit_stds[idx] = float(raw_np.std())
                logit_mins[idx] = float(raw_np.min())
                logit_maxs[idx] = float(raw_np.max())
                bias_maxs[idx]  = float(np.abs(bias).max())
                bias_mins[idx]  = float(bias.min())          # most negative = farthest patch
                lw_vals[idx]    = module.locality_weight.item()
                ratios[idx]     = bias_maxs[idx] / (logit_stds[idx] + 1e-8)
            return hook

        for i, blk in enumerate(blocks):
            handles.append(blk.attn.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            self.model(x[:1])

        for h in handles:
            h.remove()

        # ── Print table ───────────────────────────────────────────────────
        print("\n── Locality Bias Scale Diagnostic ──────────────────────────────────")
        print(f"  Grid: {gh}×{gw}  |  N={N} patches")
        print(f"  Raw bias range (unnormalised): 0  to  -{(gh-1)**2 + (gw-1)**2}  "
              f"(= -[(G-1)²+(G-1)²])")
        print()
        header = (f"  {'Block':>5}  {'lw':>7}  {'logit_std':>10}  "
                  f"{'bias_max':>9}  {'bias_min':>9}  {'ratio':>7}  {'verdict':}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        THRESHOLDS = [(1, "weak — barely affects attention"),
                      (5, "moderate — shapes but doesn't dominate"),
                      (20, "strong — suppresses distant patches"),
                      (float("inf"), "SEVERE — distant patches are dead keys ⚠")]

        any_severe = False
        for i in range(depth):
            for thresh, label in THRESHOLDS:
                if ratios[i] < thresh:
                    verdict = label
                    break
            if ratios[i] >= 5:
                any_severe = True
            print(f"  {i+1:>5}  {lw_vals[i]:>7.4f}  {logit_stds[i]:>10.4f}  "
                  f"{bias_maxs[i]:>9.2f}  {bias_mins[i]:>9.2f}  "
                  f"{ratios[i]:>7.1f}x  {verdict}")

        print()
        if any_severe:
            print("  ⚠  One or more blocks have ratio ≥ 5.")
            print("     The locality bias likely dominates the logits, causing overfocusing.")
            print("     Recommended fix in transformer.py → _gaussian_distance_bias():")
            print("         return -dist2 / dist2.max()   # normalise to [-1, 0]")
            print("     This makes locality_weight a true [0,1] interpolation knob.")
        else:
            print("  ✓  All ratios < 5 — locality bias is not dominating.")
        print("─" * 68 + "\n")

        # ── Figure: two subplots ──────────────────────────────────────────
        x_pos = np.arange(1, depth + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, depth + 2), 8),
                                        sharex=True)

        # Top: logit std vs bias max (absolute scale)
        ax1.bar(x_pos - 0.2, logit_stds, width=0.35, label="Logit std (pre-bias)",
                color="#5C8DD6", edgecolor="white")
        ax1.bar(x_pos + 0.2, bias_maxs,  width=0.35, label="Bias max |lw × bias|",
                color="#E57373", edgecolor="white")
        ax1.set_ylabel("Magnitude")
        ax1.set_title("Attention Logit Std  vs  Locality Bias Magnitude\n"
                      "If red bars >> blue bars → bias dominates → overfocusing",
                      fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

        # Bottom: ratio with threshold bands
        ratio_colors = []
        for r in ratios:
            if r < 1:
                ratio_colors.append("#4CAF50")    # green  — weak
            elif r < 5:
                ratio_colors.append("#FFC107")    # amber  — moderate
            elif r < 20:
                ratio_colors.append("#FF7043")    # orange — strong
            else:
                ratio_colors.append("#D32F2F")    # red    — severe

        bars2 = ax2.bar(x_pos, ratios, color=ratio_colors, edgecolor="white")
        ax2.axhline(1,  color="#4CAF50", linewidth=1.2, linestyle="--",
                    label="1× — bias becomes noticeable")
        ax2.axhline(5,  color="#FFC107", linewidth=1.2, linestyle="--",
                    label="5× — bias strongly shapes attention")
        ax2.axhline(20, color="#D32F2F", linewidth=1.2, linestyle="--",
                    label="20× — severe overfocusing")
        for bar, val in zip(bars2, ratios):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                     f"{val:.1f}×", ha="center", va="bottom", fontsize=8)
        ax2.set_xlabel("Block")
        ax2.set_ylabel("Ratio  bias_max / logit_std")
        ax2.set_title("Bias / Logit Ratio per Block  (higher = more overfocusing risk)",
                      fontweight="bold")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"Block {i+1}" for i in range(depth)])
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self._savefig(fig, filename)

    # ── 6. Prediction error map ──────────────────────────────────────────────

    def plot_prediction_error_map(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        filename: str = "prediction_error_map.png",
    ):
        """
        For one sample, show: input | prediction | ground truth | error map.

        Parameters
        ----------
        x      : (1, 1, 457, 457) input tensor
        y_true : (1, 1, 457, 457) ground truth tensor
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x[:1]).cpu()
        x_np     = x[0, 0].cpu().numpy()
        y_np     = y_true[0, 0].cpu().numpy()
        y_pred_np = y_pred[0, 0].numpy()
        err_np   = np.abs(y_pred_np - y_np)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ["Input (t)", "Prediction (t+1)", "Ground Truth (t+1)", "Absolute Error"]
        arrays = [x_np, y_pred_np, y_np, err_np]
        cmaps  = ["RdYlBu_r", "RdYlBu_r", "RdYlBu_r", "hot"]

        for ax, title, arr, cmap in zip(axes, titles, arrays, cmaps):
            im = ax.imshow(arr, cmap=cmap, interpolation="nearest")
            ax.set_title(title, fontweight="bold")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        mse = float(((y_pred_np - y_np) ** 2).mean())
        fig.suptitle(f"Sample Prediction  |  MSE = {mse:.6f}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        self._savefig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone: fold training summary
# ──────────────────────────────────────────────────────────────────────────────

def plot_fold_summary(
    all_fold_history: List[Dict],
    save_path: str = "fold_summary.png",
):
    """
    Comprehensive summary plot across all folds.

    Row 1: per-fold train vs val MSE curves
    Row 2: per-fold train vs val R² curves
    Row 3: final-epoch MSE and R² bar charts (fold comparison)
    """
    n_folds = len(all_fold_history)
    cmap = plt.cm.tab10

    fig = plt.figure(figsize=(max(18, 3 * n_folds), 12))
    gs = gridspec.GridSpec(3, n_folds, figure=fig, hspace=0.45, wspace=0.3)

    # Rows 1 & 2: per-fold curves
    for i, fh in enumerate(all_fold_history):
        epochs = np.arange(1, len(fh["train_mse"]) + 1)

        # MSE
        ax_mse = fig.add_subplot(gs[0, i])
        ax_mse.plot(epochs, fh["train_mse"], alpha=0.55, color=cmap(i), linewidth=1.2,
                    label="Train")
        ax_mse.plot(epochs, fh["val_mse"], color=cmap(i), linewidth=1.8,
                    linestyle="--", label="Val")
        ax_mse.set_title(f"Fold {i+1}", fontsize=9, fontweight="bold")
        if i == 0:
            ax_mse.set_ylabel("MSE", fontsize=8)
        ax_mse.tick_params(labelsize=7)
        ax_mse.legend(fontsize=6)

        # R²
        ax_r2 = fig.add_subplot(gs[1, i])
        r2_train = [t.item() if hasattr(t, "item") else t for t in fh["train_r2"]]
        r2_val   = [t.item() if hasattr(t, "item") else t for t in fh["val_r2"]]
        ax_r2.plot(epochs, r2_train, alpha=0.55, color=cmap(i), linewidth=1.2)
        ax_r2.plot(epochs, r2_val,   color=cmap(i), linewidth=1.8, linestyle="--")
        ax_r2.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        if i == 0:
            ax_r2.set_ylabel("R²", fontsize=8)
        ax_r2.set_xlabel("Epoch", fontsize=7)
        ax_r2.tick_params(labelsize=7)

    # Row 3: final epoch comparison bar charts
    ax_bar_mse = fig.add_subplot(gs[2, : n_folds // 2])
    ax_bar_r2  = fig.add_subplot(gs[2, n_folds // 2 :])

    fold_labels = [f"F{i+1}" for i in range(n_folds)]
    final_val_mse = [fh["val_mse"][-1] for fh in all_fold_history]
    final_val_r2  = [
        (fh["val_r2"][-1].item() if hasattr(fh["val_r2"][-1], "item") else fh["val_r2"][-1])
        for fh in all_fold_history
    ]

    best_mse_idx = int(np.argmin(final_val_mse))
    best_r2_idx  = int(np.argmax(final_val_r2))

    bar_colors_mse = [
        "#FF5722" if j == best_mse_idx else "#90A4AE" for j in range(n_folds)
    ]
    bar_colors_r2 = [
        "#4CAF50" if j == best_r2_idx else "#90A4AE" for j in range(n_folds)
    ]

    ax_bar_mse.bar(fold_labels, final_val_mse, color=bar_colors_mse, edgecolor="white")
    ax_bar_mse.set_title("Final Val MSE per Fold\n(orange = best)", fontweight="bold", fontsize=9)
    ax_bar_mse.set_ylabel("MSE")
    ax_bar_mse.tick_params(labelsize=8)
    ax_bar_mse.grid(axis="y", alpha=0.3)

    ax_bar_r2.bar(fold_labels, final_val_r2, color=bar_colors_r2, edgecolor="white")
    ax_bar_r2.set_title("Final Val R² per Fold\n(green = best)", fontweight="bold", fontsize=9)
    ax_bar_r2.set_ylabel("R²")
    ax_bar_r2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax_bar_r2.tick_params(labelsize=8)
    ax_bar_r2.grid(axis="y", alpha=0.3)

    fig.suptitle("Multi-Fold Cross-Validation Summary", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Quick integration snippet (paste into main.py after training)
# ──────────────────────────────────────────────────────────────────────────────

INTEGRATION_SNIPPET = """
# ── Add to main.py after diff_model_multi_fold_cv_train_test() ──────────────

from model_interpretability import ModelInterpreter, plot_fold_summary
from transformer import SmallDataDecoderViT

# 1. Training summary across all folds
plot_fold_summary(all_fold_history, save_path="fold_summary.png")

# 2. Load the best model
best_model = SmallDataDecoderViT(
    in_channels=1, embed_dim=192, depth=6, num_heads=3,
    proj_drop=0.1, drop_path_rate=0.05,
)
best_model.load_state_dict(torch.load(model_path, map_location="cpu"))

interp = ModelInterpreter(best_model, save_dir=".")

# 3. Get one sample from the last val_loader fold (rebuild quickly)
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit

X = distance_matrix[:-1][:, np.newaxis, :]
y = distance_matrix[1:][:, np.newaxis, :]
X_t = torch.from_numpy(X).float()
y_t = torch.from_numpy(y).float()

tscv = TimeSeriesSplit(n_splits=9, max_train_size=504, test_size=126)
*_, (_, last_val_idx) = tscv.split(X_t)
sample_x = X_t[last_val_idx[:1]]   # (1, 1, 457, 457)
sample_y = y_t[last_val_idx[:1]]

# 4. Generate all interpretation plots
interp.plot_attention_maps(sample_x, layer=0)
interp.plot_attention_maps(sample_x, layer=best_model.blocks.__len__() - 1,
                           filename="attention_maps_last_block.png")
interp.plot_mean_attention_distance(sample_x)
interp.plot_layerscale_gammas()
interp.plot_attention_temperatures()
interp.plot_locality_weights()
interp.plot_locality_bias_scale(sample_x)   # ← diagnoses overfocusing risk
interp.plot_prediction_error_map(sample_x, sample_y)
"""

if __name__ == "__main__":
    print(INTEGRATION_SNIPPET)
