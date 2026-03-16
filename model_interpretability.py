"""
model_interpretability.py
=========================
Extract and visualise interpretability signals from SmallDataDecoderViT
with Sector-Gated Positional Self-Attention (SectorGPSA).

Public API
----------
    ModelInterpreter
        .plot_attention_maps(x, layer, query_patch, filename)
            Per-head effective attention maps (weighted combination of
            positional and content components according to the gate).

        .plot_attention_maps_overlay(x, layer, query_patch, filename)
            Colour-coded multi-head overlay of effective attention.

        .plot_gate_values()
            Heatmap of learned gate g = sigmoid(λ) per (block, head).
            g ≈ 1 → head is nearly fully positional (sector-averaged).
            g ≈ 0 → head is nearly fully content-driven.

        .plot_mean_attention_distance(x)
            Mean spatial distance of effective attention per (block, head),
            compared against the uniform and pure-positional baselines.

        .plot_layerscale_gammas()
            LayerScale γ values (residual branch health).

        .plot_attention_weights(x)
            Content attention weight distributions (pre-gate) per block,
            useful for checking entropy / sharpness of the content stream.

        .plot_prediction_error_map(x, y_true, tickers, sector_boundaries,
                                   X_val, y_val, baseline_pred,
                                   baseline_val_full, baseline_name)
            Input D_t | Predicted ΔD̂_t | Ground Truth ΔD_t |
            Signed Error (ΔD̂−ΔD) | Per-pixel Skill Score (ΔD̂−ΔD)²/ΔD²
            Panel 4: direction of mistakes (red=over, blue=under).
            Panel 5: skill score — green<1 beats null, red>1 worse, grey=|ΔD|≈0.
            All y tensors are scaled ΔD (change), not scaled D (level).
            baseline_pred / baseline_val_full must be the null-model prediction
            in scaled space (= −y_mean/y_std), supplied from main.py.

    plot_fold_summary(all_fold_history, save_path)
        Multi-fold CV training summary (MSE + R² curves and bar charts).

Removed (LSA-specific, no longer applicable)
---------------------------------------------
    plot_attention_temperatures  — LSA used a per-head temperature scalar;
                                   SectorGPSA uses fixed scale = 1/√d.
    plot_locality_weights        — LSA's Gaussian locality bias weight;
                                   replaced by the gate in SectorGPSA.
    plot_locality_bias_scale     — diagnostic for LSA's bias/logit ratio;
                                   not meaningful for SectorGPSA.

Usage
-----
    from transformer import SmallDataDecoderViT
    from model_interpretability import ModelInterpreter, plot_fold_summary

    model = SmallDataDecoderViT(..., sector_ids=sector_ids)
    model.load_state_dict(torch.load("model_fold_9.pth", map_location="cpu"))

    interp = ModelInterpreter(model)
    sample_x = ...   # (1, 1, 457, 457) — from the validation set
    sample_y = ...   # (1, 1, 457, 457)
    X_val    = ...   # (N_val, 1, 457, 457) — full validation split
    y_val    = ...   # (N_val, 1, 457, 457)

    interp.plot_attention_maps(sample_x, layer=0)
    interp.plot_gate_values()
    interp.plot_mean_attention_distance(sample_x)
    interp.plot_layerscale_gammas()
    interp.plot_attention_weights(sample_x)
    interp.plot_prediction_error_map(sample_x, sample_y,
                                     tickers=tickers_gics,
                                     sector_boundaries=sector_boundaries,
                                     X_val=X_val, y_val=y_val,
                                     baseline_pred=baseline_pred,       # torch.full_like(sample_y, -y_mean/y_std)
                                     baseline_val_full=baseline_val_full, # torch.full_like(y_val, -y_mean/y_std)
                                     baseline_name="zero (ΔD=0)")
    plot_fold_summary(all_fold_history)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Hook helper
# ──────────────────────────────────────────────────────────────────────────────

class _GPSAHook:
    """
    Registers a forward hook on a SectorGPSA layer to capture:
        - effective_attn : (B, H, N, N) gated attention weights
          = g_h * A_pos  +  (1 - g_h) * A_content
        - content_attn   : (B, H, N, N) content-only attention weights
        - gate_values    : (H,) gate scalars g = sigmoid(lambda)

    Implementation note
    -------------------
    The hook re-derives attention weights from the module's inputs and
    parameters rather than intercepting intermediate tensors from forward().
    This avoids in-place modifications and custom autograd functions that
    break torch.compile.  The computation exactly matches SectorGPSA.forward():

        qkv = module.qkv(x).reshape(B, N, 3, H, D)
        q, k, v = permute/unbind
        A_content = softmax((q @ k.transpose(-2,-1)) * scale)
        A_pos = module._a_pos  (pre-built buffer)
        g = sigmoid(module.gate_logit)
        effective = g * A_pos + (1-g) * A_content

    The hook uses einsum("mn,bhnd->bhmd") for the positional matmul, matching
    the forward implementation.  If SectorGPSA.forward() changes, update this
    hook to maintain consistency.
    """

    def __init__(self):
        self.effective_attn: Optional[torch.Tensor] = None
        self.content_attn:   Optional[torch.Tensor] = None
        self.gate_values:    Optional[torch.Tensor] = None
        self._handle = None

    def register(self, layer):
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inputs, output):
        x = inputs[0]
        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # Content attention (no dropout for deterministic visualisation)
        scale  = module.head_dim ** -0.5
        a_cont = (q @ k.transpose(-2, -1)) * scale
        a_cont = a_cont.softmax(dim=-1)                # (B, H, N, N)

        # Positional attention — read the pre-built buffer directly.
        # _a_pos is registered as a non-persistent buffer in SectorGPSA.__init__
        # via register_buffer("_a_pos", ..., persistent=False).
        # It is always present after construction and moves with .to(device).
        A_pos   = module._a_pos                        # (N, N)
        A_pos_e = A_pos.unsqueeze(0).unsqueeze(0)      # (1, 1, N, N)

        # Gate
        g   = module.gate_logit.sigmoid()              # (H,)
        g_e = g.view(1, module.num_heads, 1, 1)        # (1, H, 1, 1)

        effective = g_e * A_pos_e + (1.0 - g_e) * a_cont  # (B, H, N, N)

        self.effective_attn = effective.detach().cpu()
        self.content_attn   = a_cont.detach().cpu()
        self.gate_values    = g.detach().cpu()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _to_float(val) -> float:
    """Safely convert a tensor scalar or plain Python float to float."""
    return val.item() if hasattr(val, "item") else float(val)


# ──────────────────────────────────────────────────────────────────────────────
# Main interpreter class
# ──────────────────────────────────────────────────────────────────────────────

class ModelInterpreter:
    """
    Interpretability toolkit for SmallDataDecoderViT with SectorGPSA.

    Parameters
    ----------
    model    : SmallDataDecoderViT — trained model instance.
    save_dir : str — directory where figures are saved (default: current dir).
    """

    def __init__(self, model, save_dir: str = "."):
        self.model    = model
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

    def _get_hook_data(self, x: torch.Tensor, layer: int) -> _GPSAHook:
        """Run a forward pass with a hook on the given layer, return hook."""
        hook = _GPSAHook()
        hook.register(self._blocks()[layer].attn)
        with torch.no_grad():
            self.model(x[:1])
        hook.remove()
        return hook

    # ── 1. Effective attention maps ───────────────────────────────────────────

    def plot_attention_maps(
        self,
        x: torch.Tensor,
        layer: int = 0,
        query_patch: Optional[int] = None,
        filename: str = "attention_maps.png",
    ):
        """
        Plot per-head effective attention maps for a given layer.

        Effective attention = g_h * A_pos  +  (1 - g_h) * A_content.
        For each head shows the distribution from *query_patch* to all
        other patches, reshaped to the 2-D grid.

        The gate value g_h is annotated on each subplot title so the
        positional/content balance is immediately visible.

        Parameters
        ----------
        x            : Input tensor (B, C, H, W).
        layer        : Transformer block index (0-indexed).
        query_patch  : Source patch index.  Default = centre patch.
        filename     : Output filename.
        """
        gh, gw = self._grid()
        N      = gh * gw
        if query_patch is None:
            query_patch = N // 2

        hook = self._get_hook_data(x, layer)
        attn = hook.effective_attn   # (1, H, N, N)
        gates = hook.gate_values     # (H,)
        H = attn.shape[1]

        ncols = min(H, 4)
        nrows = math.ceil(H / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()
        fig.suptitle(
            f"Effective Attention Maps — Block {layer+1}  |  "
            f"Query patch {query_patch}  ({gh}×{gw} grid)\n"
            f"Effective = g·A_pos + (1−g)·A_content",
            fontsize=11, fontweight="bold",
        )

        qy, qx = divmod(query_patch, gw)

        for h in range(H):
            ax = axes[h]
            weights = attn[0, h, query_patch, :].numpy()
            grid_img = weights.reshape(gh, gw)
            im = ax.imshow(grid_img, cmap="viridis", interpolation="nearest")
            g_val = gates[h].item()
            ax.set_title(
                f"Head {h+1}  g={g_val:.3f}\n"
                f"({'positional' if g_val > 0.5 else 'content'})",
                fontsize=8,
            )
            ax.axis("off")
            ax.scatter([qx], [qy], c="red", s=60, marker="x", zorder=5)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes[H:]:
            ax.axis("off")

        plt.tight_layout()
        self._savefig(fig, filename)

    # ── 1b. Colour-coded attention overlay ───────────────────────────────────

    def plot_attention_maps_overlay(
        self,
        x: torch.Tensor,
        layer: int = 0,
        query_patch: Optional[int] = None,
        filename: str = "attention_maps_overlay.png",
    ):
        """
        Plot all heads on a single colour-blended map using effective attention.

        Each head is assigned a distinct hue; its effective attention weights
        from *query_patch* are used as the alpha (opacity) channel.
        A companion figure shows each head's colour-masked contribution
        individually.

        Parameters
        ----------
        x            : Input tensor (B, C, H, W).
        layer        : Transformer block index (0-indexed).
        query_patch  : Source patch index.  Default = centre patch.
        filename     : Output filename for the blended overlay.
                       Individual-head figure is saved as 'ind_' + filename.
        """
        gh, gw = self._grid()
        N      = gh * gw
        if query_patch is None:
            query_patch = N // 2

        hook  = self._get_hook_data(x, layer)
        attn  = hook.effective_attn   # (1, H, N, N)
        gates = hook.gate_values      # (H,)
        H     = attn.shape[1]

        import matplotlib.colors as mcolors
        hues     = np.linspace(0, 1, H, endpoint=False)
        head_rgb = np.array([mcolors.hsv_to_rgb([h, 0.85, 0.95]) for h in hues])

        qy, qx = divmod(query_patch, gw)

        head_rgba = []
        for h in range(H):
            w      = attn[0, h, query_patch, :].numpy()
            w_norm = w / (w.max() + 1e-8)
            alpha  = w_norm.reshape(gh, gw)
            rgba   = np.zeros((gh, gw, 4), dtype=np.float32)
            rgba[..., :3] = head_rgb[h]
            rgba[..., 3]  = alpha
            head_rgba.append(rgba)

        # Alpha-composite onto white background
        composite = np.ones((gh, gw, 3), dtype=np.float32)
        for rgba in head_rgba:
            src_a = rgba[..., 3:4]
            src_rgb = rgba[..., :3]
            composite = src_rgb * src_a + composite * (1.0 - src_a)
        composite = np.clip(composite, 0, 1)

        # Figure 1: blended overlay
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(composite, interpolation="nearest", origin="upper")
        ax.scatter([qx], [qy], c="black", s=120, marker="x",
                   linewidths=2, zorder=5)
        ax.set_title(
            f"Effective Attention Overlay — Block {layer+1}  |  "
            f"Query patch {query_patch}\n"
            f"Colour = head  ·  Brightness = effective attention strength",
            fontsize=9, fontweight="bold",
        )
        ax.axis("off")

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(
                facecolor=head_rgb[h], edgecolor="grey",
                label=f"Head {h+1}  g={gates[h].item():.2f}",
            )
            for h in range(H)
        ]
        legend_handles.append(
            plt.Line2D([0], [0], marker="x", color="black", linestyle="None",
                       markersize=8, markeredgewidth=2, label="Query patch")
        )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
                  framealpha=0.85, edgecolor="grey")

        plt.tight_layout()
        self._savefig(fig, filename)

        # Figure 2: individual head colour masks
        ncols = min(H, 4)
        nrows = math.ceil(H / ncols)
        fig2, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()
        fig2.suptitle(
            f"Per-Head Effective Attention Colour Masks — Block {layer+1}  |  "
            f"Query patch {query_patch}",
            fontsize=10, fontweight="bold",
        )

        for h in range(H):
            ax2    = axes[h]
            single = np.ones((gh, gw, 3), dtype=np.float32)
            src_a  = head_rgba[h][..., 3:4]
            src_rgb = head_rgba[h][..., :3]
            single  = src_rgb * src_a + single * (1.0 - src_a)
            single  = np.clip(single, 0, 1)

            ax2.imshow(single, interpolation="nearest", origin="upper")
            ax2.scatter([qx], [qy], c="black", s=80, marker="x",
                        linewidths=1.5, zorder=5)
            g_val = gates[h].item()
            ax2.set_title(
                f"Head {h+1}  g={g_val:.3f}  "
                f"({'pos' if g_val > 0.5 else 'content'})",
                fontsize=9, fontweight="bold",
                color=head_rgb[h] * 0.6,
            )
            ax2.axis("off")

        for ax2 in axes[H:]:
            ax2.axis("off")

        plt.tight_layout()
        self._savefig(fig2, "ind_" + filename)

    # ── 2. Gate values ────────────────────────────────────────────────────────

    def plot_gate_values(self, filename: str = "gate_values.png"):
        """
        Heatmap of learned gate g = sigmoid(λ) per (block × head).

        g ≈ 1  → head relies almost entirely on the sectoral positional prior
                 (uniform attention within GICS sector).
        g ≈ 0  → head is almost entirely content-driven (standard QK attention).
        0 < g < 1 → interpolation; the model has learned how much domain prior
                    to blend for each head at each depth.

        Also prints a per-block, per-head summary table to stdout.
        """
        blocks = self._blocks()
        depth  = len(blocks)
        H      = blocks[0].attn.num_heads

        gates = np.zeros((depth, H))
        for l, blk in enumerate(blocks):
            gates[l] = blk.attn.gate_logit.sigmoid().detach().cpu().numpy()

        # ── Print table ───────────────────────────────────────────────────
        print("\n── Gate Values g = sigmoid(λ) ──────────────────────────────────")
        print(f"  {'Block':>5}  " + "  ".join(f"Head{h+1:>2}" for h in range(H)))
        print("  " + "-" * (7 + 9 * H))
        for l in range(depth):
            row = "  ".join(f"{gates[l, h]:>6.3f}" for h in range(H))
            print(f"  {l+1:>5}  {row}")
        print(f"\n  g≈1 = positional (sector prior)  |  g≈0 = content-driven\n")

        # ── Figure ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(5, H + 1), depth + 1))
        im = ax.imshow(gates, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(H))
        ax.set_xticklabels([f"H{h+1}" for h in range(H)])
        ax.set_yticks(range(depth))
        ax.set_yticklabels([f"Block {l+1}" for l in range(depth)])
        ax.set_xlabel("Attention Head")
        ax.set_title(
            "SectorGPSA Gate Values  g = sigmoid(λ)  per Block × Head\n"
            "Green (g→1) = positional (sector prior)  |  "
            "Red (g→0) = content-driven",
            fontweight="bold",
        )
        fig.colorbar(im, ax=ax, label="gate g ∈ (0, 1)")

        for l in range(depth):
            for h in range(H):
                val = gates[l, h]
                text_col = "white" if val < 0.25 or val > 0.75 else "black"
                ax.text(h, l, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=text_col, fontweight="bold")

        plt.tight_layout()
        self._savefig(fig, filename)

    # ── 3. Mean attention distance ───────────────────────────────────────────

    def plot_mean_attention_distance(
        self,
        x: torch.Tensor,
        filename: str = "mean_attention_distance.png",
    ):
        """
        Mean spatial distance (patch hops) of effective attention per (layer, head).

        Baselines computed:
          - Uniform baseline  : mean distance under fully uniform attention over N.
          - Positional baseline: mean distance of pure sector attention (g=1).
            Because A_pos attends uniformly within sectors, this equals the
            mean intra-sector pairwise distance, which is lower than uniform.

        Produces two figures:
          1. Heatmap  (depth × num_heads)       [filename]
          2. Bar chart per layer with baselines  ['bar_' + filename]
        """
        gh, gw = self._grid()
        N      = gh * gw
        depth  = len(self._blocks())

        # ── Patch coordinates and full pairwise distance matrix ───────────
        gy  = torch.arange(gh).float()
        gx  = torch.arange(gw).float()
        cy, cx = torch.meshgrid(gy, gx, indexing="ij")
        coords  = torch.stack([cy.flatten(), cx.flatten()], dim=-1)   # (N, 2)
        dist_mat = torch.cdist(coords, coords, p=2).numpy()           # (N, N)

        # ── Uniform baseline ──────────────────────────────────────────────
        uniform_baseline = dist_mat.mean()

        # ── Positional baseline (pure sector attention, g=1) ──────────────
        # Use the pre-built _a_pos buffer directly — consistent with forward()
        A_pos_np = (
            self._blocks()[0]
            .attn._a_pos
            .cpu()
            .numpy()
        )   # (N, N)
        positional_baseline = (A_pos_np * dist_mat).sum(axis=-1).mean()

        print(f"\n── Mean Attention Distance Baselines (grid={gh}×{gw}, N={N}) ──")
        print(f"  Uniform attention baseline   : {uniform_baseline:.4f}")
        print(f"  Pure positional (g=1) baseline: {positional_baseline:.4f}  "
              f"(mean intra-sector distance)")
        print(f"  → Effective distance < positional baseline  "
              f"⟹ content head is also local")
        print(f"  → Effective distance between baselines  "
              f"⟹ gate is blending meaningfully\n")

        # ── Compute per-(layer, head) mean attention distance ──────────────
        num_heads = self._blocks()[0].attn.num_heads
        mean_dist = np.zeros((depth, num_heads))

        for layer in range(depth):
            hook = self._get_hook_data(x, layer)
            attn = hook.effective_attn   # (1, H, N, N)
            for h in range(num_heads):
                w = attn[0, h].numpy()
                mean_dist[layer, h] = (w * dist_mat).sum(axis=-1).mean()

        # ── Figure 1: heatmap ──────────────────────────────────────────────
        H = num_heads
        fig, ax = plt.subplots(figsize=(max(6, H), depth + 1))
        im = ax.imshow(mean_dist, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=dist_mat.max())
        ax.set_xticks(range(H))
        ax.set_xticklabels([f"H{h+1}" for h in range(H)])
        ax.set_yticks(range(depth))
        ax.set_yticklabels([f"Block {l+1}" for l in range(depth)])
        ax.set_xlabel("Attention Head")
        ax.set_title(
            f"Mean Effective Attention Distance (patch hops)  —  grid {gh}×{gw}\n"
            f"Uniform={uniform_baseline:.2f}  |  "
            f"Positional (g=1)={positional_baseline:.2f}  |  "
            f"Max={dist_mat.max():.2f}",
            fontweight="bold", fontsize=9,
        )
        fig.colorbar(im, ax=ax, label="mean hop distance")

        for l in range(depth):
            for h in range(H):
                val = mean_dist[l, h]
                text_col = "white" if val > dist_mat.max() * 0.6 else "black"
                ax.text(h, l, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=text_col, fontweight="bold")

        plt.tight_layout()
        self._savefig(fig, filename)

        # ── Figure 2: bar chart per layer with both baselines ─────────────
        mean_per_layer = mean_dist.mean(axis=1)
        bar_filename   = "bar_" + filename

        fig2, ax2 = plt.subplots(figsize=(max(7, depth + 2), 4))
        x_pos = np.arange(1, depth + 1)
        bars  = ax2.bar(x_pos, mean_per_layer, color="#5C8DD6",
                        edgecolor="white", linewidth=0.6,
                        label="Model (mean over heads)")

        ax2.axhline(uniform_baseline, color="#E53935", linewidth=1.5,
                    linestyle="--",
                    label=f"Uniform baseline ({uniform_baseline:.2f})")
        ax2.axhline(positional_baseline, color="#43A047", linewidth=1.5,
                    linestyle=":",
                    label=f"Positional baseline g=1 ({positional_baseline:.2f})")
        ax2.axhline(0, color="black", linewidth=0.6)

        for bar, val in zip(bars, mean_per_layer):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax2.set_xlabel("Block")
        ax2.set_ylabel("Mean attention distance (patch hops)")
        ax2.set_title(
            "Mean Effective Attention Distance per Block\n"
            "Below positional baseline → content stream is also local  |  "
            "Above uniform → more global than chance",
            fontweight="bold",
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"Block {l+1}" for l in range(depth)])
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, dist_mat.max() * 1.1)
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._savefig(fig2, bar_filename)

    # ── 4. LayerScale gammas ─────────────────────────────────────────────────

    def plot_layerscale_gammas(self, filename: str = "layerscale_gammas.png"):
        """
        Plot the LayerScale γ values for each block.

        γ ≈ 0  → residual branch suppressed (block barely contributes).
        γ ≈ 1  → full-strength residual.

        Two lines per block: attention branch (ls1) and FFN branch (ls2).
        """
        blocks = self._blocks()
        depth  = len(blocks)

        ls1_means, ls1_stds = [], []
        ls2_means, ls2_stds = [], []

        for blk in blocks:
            g1 = blk.ls1.gamma.detach().cpu().numpy()
            g2 = blk.ls2.gamma.detach().cpu().numpy()
            ls1_means.append(g1.mean()); ls1_stds.append(g1.std())
            ls2_means.append(g2.mean()); ls2_stds.append(g2.std())

        x_pos = np.arange(1, depth + 1)
        fig, ax = plt.subplots(figsize=(max(7, depth + 1), 4))
        ax.errorbar(x_pos, ls1_means, yerr=ls1_stds, marker="o",
                    label="Attention branch (ls1)",
                    color="#2196F3", capsize=4)
        ax.errorbar(x_pos, ls2_means, yerr=ls2_stds, marker="s",
                    label="FFN branch (ls2)",
                    color="#F44336", capsize=4, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Block")
        ax.set_ylabel("LayerScale γ (mean ± std)")
        ax.set_title(
            "LayerScale Gammas per Block\n"
            "γ ≈ 0 → branch suppressed  |  γ ≈ 1 → full strength",
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        ax.legend()
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._savefig(fig, filename)

    # ── 5. Content attention weight distribution ──────────────────────────────

    def plot_attention_weights(
        self,
        x: torch.Tensor,
        filename: str = "attention_weights.png",
    ):
        """
        Visualise the content attention weight distributions per block.

        For each block, shows the distribution of A_content values across
        all (query, key) pairs and heads.  This is the pre-gate stream and
        indicates how sharp or diffuse the content component is, independent
        of the gate.

        A very peaked distribution → content stream attends sharply to a
        few tokens (low entropy, high confidence).
        A flat distribution → content stream is diffuse / uncertain.

        Produced as a grid of violin plots, one per block.
        """
        depth  = len(self._blocks())
        H      = self._blocks()[0].attn.num_heads

        all_content = []   # list of (depth,) each entry is flattened (H*N*N,)

        for layer in range(depth):
            hook = self._get_hook_data(x, layer)
            # content_attn shape: (1, H, N, N)
            vals = hook.content_attn[0].numpy().flatten()   # (H*N*N,)
            all_content.append(vals)

        fig, axes = plt.subplots(
            1, depth, figsize=(3 * depth, 4), sharey=True
        )
        if depth == 1:
            axes = [axes]

        for l, (ax, vals) in enumerate(zip(axes, all_content)):
            ax.violinplot([vals], positions=[0], showmedians=True,
                          showextrema=True)
            ax.set_title(f"Block {l+1}", fontsize=9)
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.5)
            if l == 0:
                ax.set_ylabel("Content attention weight")

            # Annotate median and entropy
            median_val = float(np.median(vals))
            # Shannon entropy of normalised histogram
            counts, bin_edges = np.histogram(vals, bins=50, density=True)
            bin_w = bin_edges[1] - bin_edges[0]
            probs = counts * bin_w
            probs = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            ax.text(0, ax.get_ylim()[1] * 0.95,
                    f"med={median_val:.4f}\nH={entropy:.2f}",
                    ha="center", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        fig.suptitle(
            "Content Attention Weight Distributions per Block\n"
            "(pre-gate A_content; H = Shannon entropy of distribution)",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        self._savefig(fig, filename)

    # ── 6. Prediction error map ──────────────────────────────────────────────

    def plot_prediction_error_map(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        filename: str = "prediction_error_map.png",
        tickers: Optional[List[str]] = None,
        sector_boundaries: Optional[List[Tuple[str, int, int]]] = None,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        baseline_pred: Optional[torch.Tensor] = None,
        baseline_name: str = "persistence",
        baseline_val_full: Optional[torch.Tensor] = None,
    ):
        """
        For one sample, show five panels:
          1. Input  D_t  (scaled level — what the model sees)
          2. Prediction  ΔD̂_t  = model(D_t)  (predicted change)
          3. Ground truth  ΔD_t  = D_{t+1} − D_t  (actual change)
          4. Signed Error  ΔD̂_t − ΔD_t
               red   (> 0) : model over-predicts the change
               blue  (< 0) : model under-predicts the change
               white (= 0) : exact prediction
             Symmetric RdBu_r colourmap centred at 0.  Shows the *direction*
             of mistakes, which is invisible in any absolute/squared error map.
          5. Per-pixel Skill Score  (ΔD̂_t − ΔD_t)² / ΔD_t²
               = per-pixel contribution to the Skill Score training loss.
               < 1 (green) : model error < null error — beats null at this cell
               = 1 (white) : model ties the null (dashed line on colourbar)
               > 1 (red)   : model error > null error — worse than null here
               grey        : |ΔD_t| < 5th percentile — masked, carries no loss weight

        The model predicts the *change* ΔD_t = D_{t+1} − D_t, not the next
        level D_{t+1}.  All y tensors passed here must therefore be scaled ΔD,
        not scaled D.

        When tickers and sector_boundaries are supplied, GICS sector divider
        lines and sector name labels are drawn on every panel.

        Null model
        ----------
        The null model predicts ΔD = 0 (no change in raw space).  In scaled
        space this corresponds to the constant −y_mean_train / y_std_train,
        NOT zero.  The caller (main.py) must supply this as `baseline_pred`
        (single sample) and `baseline_val_full` (full val set).

        Full-Validation MSE Comparison
        ------------------------------
        When X_val / y_val are supplied (the full validation split), both the
        model MSE and the null-model MSE are computed over all validation
        samples, matching the val_mse metric reported during training.

        Notes on the MSE denominator
        ----------------------------
        The distance matrix is symmetric (D[i,j] == D[j,i]) and the diagonal
        is constant across time steps, so both facts affect model and null MSE
        equally — the relative improvement % is unbiased.  The effective number
        of independent observations is 457×456/2 = 104,196 (upper triangle,
        excluding diagonal), not 457² = 208,849.

        Parameters
        ----------
        x                 : (1, 1, 457, 457) input D_t tensor (scaled, GICS-reordered).
        y_true            : (1, 1, 457, 457) ground truth ΔD_t tensor (scaled).
        filename          : Output filename.
        tickers           : list[str] of 457 ticker labels in GICS order.
        sector_boundaries : list of (sector_name, start_idx, end_idx) tuples.
                            end_idx is exclusive.
        X_val             : (N_val, 1, 457, 457) full validation D_t inputs (scaled).
                            When supplied, MSE stats use the full val set.
        y_val             : (N_val, 1, 457, 457) full validation ΔD_t targets (scaled).
                            Must be supplied together with X_val.
        baseline_pred     : (1, 1, 457, 457) null-model prediction for the single
                            displayed sample, in scaled space
                            (= torch.full_like(sample_y, −y_mean/y_std)).
        baseline_name     : str — label for the null model in plot titles.
        baseline_val_full : (N_val, 1, 457, 457) null-model prediction for the
                            full validation set, in scaled space.  Must be
                            supplied; without it the null-model MSE is wrong.
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x[:1]).cpu()

        x_np      = x[0, 0].cpu().numpy()
        y_np      = y_true[0, 0].cpu().numpy()
        y_pred_np = y_pred[0, 0].numpy()

        # ── Single-sample baseline prediction ─────────────────────────────
        # For ΔD forecasting the null model is "predict no change" (ΔD=0
        # in raw space).  In scaled space this is −y_mean/y_std, which is
        # captured by `baseline_pred` passed from main.py.  Passing x_t
        # (persistence) as a baseline is wrong here because x_t is the
        # *level* D_t, not a ΔD prediction.
        if baseline_pred is not None:
            baseline_pred_np = baseline_pred[0, 0].cpu().numpy()
        elif baseline_name == "zero":
            baseline_pred_np = np.zeros_like(y_np)
        else:
            baseline_pred_np = np.zeros_like(y_np)
        # baseline_pred_np is retained for the full-val MSE computation below

        # ── Compute MSE over the full validation set (required) ───────────
        if (X_val is None) or (y_val is None):
            raise ValueError(
                "plot_prediction_error_map requires X_val and y_val so that "
                "baseline/model MSE is computed on the full validation set "
                "(single-sample MSE is too noisy)."
            )

        mse_scope_label = f"full val set (N={len(X_val)} samples)"
        with torch.no_grad():
            y_pred_val = self.model(X_val).cpu()
        y_val_np      = y_val[:, 0].numpy()          # (N, 457, 457)
        y_pred_val_np = y_pred_val[:, 0].numpy()     # (N, 457, 457)

        # ── Null-model baseline for full validation set ───────────────────
        # For ΔD prediction the null is "predict no change" (ΔD=0 raw).
        # In scaled space this is the constant −y_mean_train/y_std_train,
        # NOT zero, because y was standardised with train-set statistics.
        # The caller (main.py) must supply baseline_val_full as a tensor of
        # shape (N_val, 1, 457, 457) filled with −y_mean/y_std.
        # If not supplied we fall back to raw zeros with a warning.
        if baseline_val_full is not None:
            baseline_val_np = baseline_val_full[:, 0].numpy()
        elif baseline_name == "zero":
            print(
                "  WARNING: baseline_val_full not supplied. Using 0 in scaled "
                "space as the null model, which is only correct when y_mean_train≈0. "
                "Pass baseline_val_full=torch.full_like(y_val, -y_mean/y_std) "
                "from main.py for an exact null-model comparison."
            )
            baseline_val_np = np.zeros_like(y_val_np)
        else:
            # Persistence baseline (level prediction): X_val as ΔD predictor
            # only makes sense if the model was NOT trained on deltas.
            print("  WARNING: using X_val (persistence) as baseline for a ΔD model. "
                  "Pass baseline_val_full for the correct zero-change null model.")
            baseline_val_np = X_val[:, 0].numpy()

        mse_model    = float(((y_pred_val_np - y_val_np) ** 2).mean())
        mse_baseline = float(((baseline_val_np - y_val_np) ** 2).mean())

        rel_improve = (
            1.0 - mse_model / mse_baseline if mse_baseline > 0.0 else 0.0
        )

        # ── Console diagnostics ───────────────────────────────────────────
        print(f"\n── Null-model baseline check ({mse_scope_label}) ──────────────")
        print(f"  Baseline: {baseline_name}  (predict ΔD=0, i.e. no change)")
        print(f"  MSE scope : {mse_scope_label}")
        print("  NOTE: MSEs are in scaled ΔD space (y_std-normalised).")
        print("        Null-model MSE ≈ 1.0 on training set by construction;")
        print("        it may differ slightly on val due to fold-specific scaling.")
        print(f"  Null-model MSE : {mse_baseline:.6e}")
        print(f"  Model MSE      : {mse_model:.6e}")
        print(f"  NOTE: D is symmetric and the diagonal is constant across")
        print(f"        time steps, so effective independent pairs = 457×456/2")
        print(f"        = 104,196 (not 457² = 208,849 as the mean suggests).")
        print(f"        The model vs null comparison remains unbiased.")

        if mse_baseline > 0.0:
            print(f"  Relative improvement vs null model : {rel_improve * 100:.2f}%")
            if mse_model < mse_baseline:
                print("  ✓ Model beats the zero-change null model.")
            else:
                print("  ✗ WARNING: model does NOT beat the null model — investigate.")
        else:
            print("  WARNING: null-model MSE = 0 — degenerate case.")
        print()

        skill_score_val = mse_model / mse_baseline if mse_baseline > 0.0 else float("nan")
        scope_short  = "val-set"
        gics_note    = " | GICS-reordered" if sector_boundaries is not None else ""
        rel_pct      = f"{rel_improve * 100:+.2f}%" if mse_baseline > 0.0 else "N/A"
        title_str    = (
            f"ΔD Prediction Sample  |  "
            f"Skill Score ({scope_short}) = {skill_score_val:.4f}  "
            f"[<1 beats null]  |  "
            f"MSE_model = {mse_model:.3e}  |  MSE_null = {mse_baseline:.3e}  |  "
            f"Improvement = {rel_pct}"
            f"{gics_note}"
        )

        # ── Panel 4: signed prediction error ─────────────────────────────
        # Shows the *direction* of the model's mistakes — lost in any absolute
        # or squared error map.  Positive (red) = model over-predicts the
        # change; negative (blue) = model under-predicts.  Symmetric colourmap
        # centred at 0 so white = exact prediction.
        signed_err_np  = y_pred_np - y_np
        max_abs_serr   = float(np.abs(signed_err_np).max()) or 1.0

        # ── Panel 5: per-pixel skill score ────────────────────────────────
        # The training loss is Skill Score = MSE_model / MSE_null, where
        # MSE_null = mean(y²).  The per-pixel contribution to this ratio is:
        #
        #   skill_pixel[i,j] = (ΔD̂[i,j] − ΔD[i,j])² / ΔD[i,j]²
        #
        # Interpretation:
        #   skill_pixel < 1  →  model error < null error at this cell (green)
        #   skill_pixel = 1  →  model ties the null (white) — null-parity line
        #   skill_pixel > 1  →  model error > null error at this cell (red)
        #
        # Cells where ΔD ≈ 0 (nearly constant pairs) are masked (grey):
        # the null error is also ≈ 0 there, the ratio diverges, and they
        # carry negligible weight in the actual loss.
        # Mask threshold = 5th percentile of |ΔD| values.
        eps_mask      = np.percentile(np.abs(y_np), 5)
        denom_safe    = np.where(np.abs(y_np) > eps_mask, y_np ** 2, np.nan)
        skill_pix_np  = (y_pred_np - y_np) ** 2 / denom_safe   # NaN where masked
        skill_vmin, skill_vmax = 0.0, 2.0

        # ── Panels ────────────────────────────────────────────────────────
        titles = [
            "Input  D_t  (scaled level)",
            "Prediction  ΔD̂_{t→t+1}",
            "Ground Truth  ΔD_{t→t+1}",
            "Signed Error  ΔD̂ − ΔD\n"
            "red = over-predict  blue = under-predict  white = exact",
            "Per-pixel Skill Score  (ΔD̂−ΔD)²/ΔD²\n"
            "<1 = beats null (green)  >1 = worse (red)  grey = |ΔD|≈0",
        ]
        arrays = [x_np, y_pred_np, y_np, signed_err_np, skill_pix_np]
        cmaps  = ["RdYlBu_r", "RdYlBu_r", "RdYlBu_r", "RdBu_r", "RdYlGn_r"]
        vmins  = [None, None, None, -max_abs_serr, skill_vmin]
        vmaxs  = [None, None, None,  max_abs_serr, skill_vmax]

        fig, axes = plt.subplots(1, 5, figsize=(30, 6))

        for ax, title, arr, cmap, vmin, vmax in zip(
            axes, titles, arrays, cmaps, vmins, vmaxs
        ):
            cmap_obj = plt.get_cmap(cmap).copy()
            if arr is skill_pix_np:
                cmap_obj.set_bad(color="lightgrey")   # masked (|ΔD|≈0) cells
            im = ax.imshow(arr, cmap=cmap_obj, interpolation="nearest",
                           vmin=vmin, vmax=vmax)
            ax.set_title(title, fontweight="bold", fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if arr is skill_pix_np:
                cb.ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--")
                cb.ax.text(
                    1.05, 1.0, "null", transform=cb.ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7, color="black",
                )

            if sector_boundaries is not None:
                for _, start, end in sector_boundaries:
                    if start > 0:
                        ax.axhline(start - 0.5, color="white",
                                   linewidth=0.8, linestyle="-", alpha=0.9)
                        ax.axvline(start - 0.5, color="white",
                                   linewidth=0.8, linestyle="-", alpha=0.9)

                mid_positions = [
                    (start + end) / 2 for _, start, end in sector_boundaries
                ]
                short_names = [
                    name.replace("Communication Services", "Comm.")
                        .replace("Consumer Discretionary", "Cons. Disc.")
                        .replace("Consumer Staples", "Cons. Stap.")
                        .replace("Information Technology", "IT")
                    for name, _, _ in sector_boundaries
                ]
                ax.set_yticks(mid_positions)
                ax.set_yticklabels(short_names, fontsize=5)
                ax.tick_params(axis="y", length=0, pad=2)
                ax.set_xticks(mid_positions)
                ax.set_xticklabels(short_names, fontsize=5,
                                   rotation=45, ha="right")
                ax.tick_params(axis="x", length=0, pad=2)
            else:
                ax.axis("off")

        fig.suptitle(title_str, fontsize=12, fontweight="bold")
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

    Note: R² values stored in fold_history are plain Python floats
    (returned by _r2_from_scalars).  _to_float() is used throughout
    to safely handle both floats and any future tensor values.
    """
    n_folds = len(all_fold_history)
    cmap    = plt.cm.tab10

    fig = plt.figure(figsize=(max(18, 3 * n_folds), 12))
    gs  = gridspec.GridSpec(3, n_folds, figure=fig, hspace=0.45, wspace=0.3)

    for i, fh in enumerate(all_fold_history):
        epochs = np.arange(1, len(fh["train_mse"]) + 1)

        ax_mse = fig.add_subplot(gs[0, i])
        ax_mse.plot(epochs, fh["train_mse"], alpha=0.55, color=cmap(i),
                    linewidth=1.2, label="Train")
        ax_mse.plot(epochs, fh["val_mse"], color=cmap(i), linewidth=1.8,
                    linestyle="--", label="Val")
        ax_mse.set_title(f"Fold {i+1}", fontsize=9, fontweight="bold")
        if i == 0:
            ax_mse.set_ylabel("MSE", fontsize=8)
        ax_mse.tick_params(labelsize=7)
        ax_mse.legend(fontsize=6)

        ax_r2 = fig.add_subplot(gs[1, i])
        r2_train = [_to_float(t) for t in fh["train_r2"]]
        r2_val   = [_to_float(t) for t in fh["val_r2"]]
        ax_r2.plot(epochs, r2_train, alpha=0.55, color=cmap(i), linewidth=1.2)
        ax_r2.plot(epochs, r2_val,   color=cmap(i), linewidth=1.8, linestyle="--")
        ax_r2.axhline(0, color="gray", linewidth=0.7, linestyle=":",
                      label="null model (ΔD=0)")
        if i == 0:
            ax_r2.set_ylabel("R² (vs zero-change null model)", fontsize=8)
        ax_r2.set_xlabel("Epoch", fontsize=7)
        ax_r2.tick_params(labelsize=7)

    mid = n_folds // 2
    ax_bar_mse = fig.add_subplot(gs[2, :mid])
    ax_bar_r2  = fig.add_subplot(gs[2, mid:])

    fold_labels   = [f"F{i+1}" for i in range(n_folds)]
    final_val_mse = [fh["val_mse"][-1] for fh in all_fold_history]
    final_val_r2  = [_to_float(fh["val_r2"][-1]) for fh in all_fold_history]

    best_mse_idx = int(np.argmin(final_val_mse))
    best_r2_idx  = int(np.argmax(final_val_r2))

    bar_colors_mse = [
        "#FF5722" if j == best_mse_idx else "#90A4AE" for j in range(n_folds)
    ]
    bar_colors_r2 = [
        "#4CAF50" if j == best_r2_idx else "#90A4AE" for j in range(n_folds)
    ]

    ax_bar_mse.bar(fold_labels, final_val_mse, color=bar_colors_mse,
                   edgecolor="white")
    ax_bar_mse.set_title("Final Val MSE per Fold\n(orange = best)",
                         fontweight="bold", fontsize=9)
    ax_bar_mse.set_ylabel("MSE")
    ax_bar_mse.tick_params(labelsize=8)
    ax_bar_mse.grid(axis="y", alpha=0.3)

    ax_bar_r2.bar(fold_labels, final_val_r2, color=bar_colors_r2,
                  edgecolor="white")
    ax_bar_r2.set_title("Final Val R² per Fold\n(green = best)",
                        fontweight="bold", fontsize=9)
    ax_bar_r2.set_ylabel("R² (vs zero-change null model)", fontsize=9)
    ax_bar_r2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax_bar_r2.tick_params(labelsize=8)
    ax_bar_r2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Multi-Fold Cross-Validation Summary — SectorGPSA\n"
        "R² > 0  ⟺  model beats the zero-change null model (ΔD=0)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)