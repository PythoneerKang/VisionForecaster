"""
gnn_cross_scale.py
==================
Temporal Graph Neural Network for cross-scale adjacency prediction.

Task
----
Given L weekly snapshots of the short-scale graph A_w35[t-L+1 : t]
(and optionally recent A_wlong history), predict A_wlong[t+1].

Architecture
------------
    Input graphs  →  GraphSAGE encoder  →  GRU temporal aggregation
                  →  pairwise inner-product decoder  →  focal loss

Why these choices (see design notes in train_cross_scale.py):
  - GraphSAGE mean aggregator: robust to the extreme sparsity of A_w35
    (mean density ~2.4%); isolated nodes get their own features, not zeros.
  - GRU over Transformer: only ~220 weekly training steps, far too few
    to train temporal self-attention reliably.
  - Focal loss γ=2–4: class imbalance 120:1 (w=35) to 478:1 (w=120);
    standard BCE collapses to all-zeros trivially.
  - Positive-pair subsampling at train time: keeps gradient concentrated
    on informative pairs without discarding any time steps.

Public API
----------
    CrossScaleGNN(...)            — the full model
    FocalLoss(gamma, pos_weight)  — loss function
    build_node_features(...)      — degree + GICS sector one-hot features

Shapes / conventions
--------------------
    N          = 457 stocks
    n_pairs    = N*(N-1)//2 = 104,196 upper-triangle pairs
    L          = history_lags (default 4 weekly steps)
    B          = 1 (no batch dimension in graph ops; time is the "batch")
    edge_index : (2, E)  LongTensor — COO format, undirected (both dirs)
    x          : (N, F)  FloatTensor — node features
"""

import math
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Node feature builder
# ─────────────────────────────────────────────────────────────────────────────

# GICS sector name → integer index (matches extract_distance_matrices.py order)
_SECTOR_TO_IDX = {
    "Communication Services":  0,
    "Consumer Discretionary":  1,
    "Consumer Staples":        2,
    "Energy":                  3,
    "Financials":              4,
    "Health Care":             5,
    "Industrials":             6,
    "Information Technology":  7,
    "Materials":               8,
    "Real Estate":             9,
    "Utilities":               10,
}
N_SECTORS = len(_SECTOR_TO_IDX)   # 11


def build_node_features(
    sector_labels: List[str],
    adj_snapshot: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Build (N, F) node feature matrix for one time step.

    Features
    --------
        Columns 0–10  : GICS sector one-hot  (11 dims)
        Column  11    : normalised degree in the current snapshot
                        (0.0 when adj_snapshot is None — useful at init)

    Parameters
    ----------
    sector_labels  : list[str] of length N, GICS-reordered sector names.
                     Must match the stock ordering used in the adjacency matrix.
    adj_snapshot   : (N, N) binary float32 array, or None.

    Returns
    -------
    torch.Tensor of shape (N, 12), dtype float32.
    """
    N = len(sector_labels)
    feats = torch.zeros(N, N_SECTORS + 1, dtype=torch.float32)

    for i, name in enumerate(sector_labels):
        idx = _SECTOR_TO_IDX.get(name, 0)
        feats[i, idx] = 1.0

    if adj_snapshot is not None:
        degree = adj_snapshot.sum(axis=1).astype(np.float32)
        max_deg = max(degree.max(), 1.0)
        feats[:, N_SECTORS] = torch.from_numpy(degree / max_deg)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Adjacency → COO edge index
# ─────────────────────────────────────────────────────────────────────────────

def adj_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    """
    Convert (N, N) binary adjacency to (2, E) LongTensor edge_index.
    Adds both directions (i→j and j→i) for undirected message passing.
    Self-loops are assumed already removed in adj.

    Returns edge_index on CPU; move to device in the training loop.
    """
    rows, cols = np.nonzero(adj)
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    return edge_index


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE mean aggregator (single layer)
# ─────────────────────────────────────────────────────────────────────────────

class SAGEConv(nn.Module):
    """
    Single GraphSAGE layer with mean aggregation.

        h_v = ReLU( W_self * x_v  +  W_neigh * mean_{u∈N(v)} x_u  +  b )

    Isolated nodes (no neighbours) receive mean = zeros, so their
    embedding is purely determined by W_self * x_v.  This is correct
    behaviour given the extreme sparsity of A_w35.

    Parameters
    ----------
    in_dim  : input feature dimension F
    out_dim : output embedding dimension
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self  = nn.Linear(in_dim,  out_dim, bias=False)
        self.lin_neigh = nn.Linear(in_dim,  out_dim, bias=False)
        self.bias      = nn.Parameter(torch.zeros(out_dim))
        self._reset()

    def _reset(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)

    def forward(
        self,
        x: torch.Tensor,          # (N, F)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:             # (N, out_dim)
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Aggregate neighbour features via mean
        # scatter_mean: for each dst node, average the src node features
        agg = torch.zeros(N, x.size(1), device=x.device, dtype=x.dtype)
        count = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, x[src])
        count.index_add_(0, dst, torch.ones(src.size(0), 1,
                                            device=x.device, dtype=x.dtype))
        # Avoid division by zero for isolated nodes
        count = count.clamp(min=1.0)
        agg = agg / count

        out = self.lin_self(x) + self.lin_neigh(agg) + self.bias
        return F.relu(out)


# ─────────────────────────────────────────────────────────────────────────────
# Two-layer GraphSAGE encoder
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGEEncoder(nn.Module):
    """
    Two-layer GraphSAGE that maps node features to node embeddings.

    Layer 1: F   → hidden_dim   (with ReLU, inside SAGEConv)
    Layer 2: hidden_dim → embed_dim  (with ReLU, inside SAGEConv)
    Final LayerNorm for stable GRU inputs.

    Parameters
    ----------
    in_dim     : input node feature dimension (12 = 11 sectors + 1 degree)
    hidden_dim : intermediate width
    embed_dim  : final node embedding dimension
    dropout    : applied between layers
    """

    def __init__(
        self,
        in_dim:     int = 12,
        hidden_dim: int = 64,
        embed_dim:  int = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.conv1   = SAGEConv(in_dim,     hidden_dim)
        self.conv2   = SAGEConv(hidden_dim, embed_dim)
        self.drop    = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,           # (N, F)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:             # (N, embed_dim)
        h = self.conv1(x, edge_index)
        h = self.drop(h)
        h = self.conv2(h, edge_index)
        return self.norm(h)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal GRU aggregator
# ─────────────────────────────────────────────────────────────────────────────

class TemporalGRU(nn.Module):
    """
    GRU that aggregates L sequential node embedding snapshots into a
    single per-node context vector.

    Input  : (L, N, embed_dim)  — sequence of node embeddings
    Output : (N, gru_dim)       — final hidden state per node

    A single GRU layer is used deliberately.  With only ~220 training
    sequences, a deeper or wider temporal model will overfit.

    Parameters
    ----------
    embed_dim : dimension of the per-step node embeddings (GRU input size)
    gru_dim   : GRU hidden state dimension (GRU output size)
    dropout   : applied to GRU output
    """

    def __init__(
        self,
        embed_dim: int = 64,
        gru_dim:   int = 64,
        dropout:   float = 0.1,
    ):
        super().__init__()
        # batch_first=False: input shape (L, N, embed_dim)
        self.gru  = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_dim,
            num_layers=1,
            batch_first=False,
            dropout=0.0,  # single layer, dropout handled externally
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(gru_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        seq : (L, N, embed_dim)

        Returns
        -------
        (N, gru_dim) — final hidden state
        """
        # GRU output: (L, N, gru_dim), h_n: (1, N, gru_dim)
        _, h_n = self.gru(seq)
        h = h_n.squeeze(0)           # (N, gru_dim)
        return self.norm(self.drop(h))


# ─────────────────────────────────────────────────────────────────────────────
# Inner-product link decoder
# ─────────────────────────────────────────────────────────────────────────────

class LinkDecoder(nn.Module):
    """
    Predict edge probabilities from node embeddings.

    For a pair (i, j):
        score(i,j) = MLP( [h_i ⊙ h_j, |h_i - h_j|] )

    The element-wise product captures multiplicative interaction;
    the absolute difference captures structural dissimilarity.
    Together they are a standard and robust pair-wise feature for
    link prediction (see Zhang & Chen, SEAL, NeurIPS 2018).

    The final sigmoid is applied in the loss function (BCEWithLogitsLoss /
    FocalLoss), NOT inside forward(), to allow numerically stable training.

    Parameters
    ----------
    gru_dim    : node embedding dimension (= GRU hidden size)
    hidden_dim : MLP hidden width
    """

    def __init__(self, gru_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gru_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        h: torch.Tensor,           # (N, gru_dim)
        pair_idx: torch.Tensor,    # (P, 2)  pair indices (i, j)
    ) -> torch.Tensor:             # (P,)   raw logits
        i_idx = pair_idx[:, 0]
        j_idx = pair_idx[:, 1]
        hi = h[i_idx]              # (P, gru_dim)
        hj = h[j_idx]              # (P, gru_dim)
        feat = torch.cat([hi * hj, (hi - hj).abs()], dim=-1)  # (P, 2*gru_dim)
        return self.mlp(feat).squeeze(-1)                       # (P,)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class CrossScaleGNN(nn.Module):
    """
    End-to-end temporal GNN for cross-scale link prediction.

    Forward pass
    ------------
    1. For each of the L input snapshots, run GraphSAGEEncoder on
       (node_features[t], edge_index_w35[t]) to get node embeddings h[t].
    2. Stack h[0..L-1] into (L, N, embed_dim) and feed to TemporalGRU.
    3. Decode selected pairs with LinkDecoder → raw logits.

    Parameters
    ----------
    in_dim         : node feature dimension (default 12)
    sage_hidden    : GraphSAGE intermediate width
    embed_dim      : GraphSAGE output / GRU input dimension
    gru_dim        : GRU hidden state dimension
    decoder_hidden : LinkDecoder MLP hidden width
    dropout        : dropout rate applied in all sub-modules
    """

    def __init__(
        self,
        in_dim:         int   = 12,
        sage_hidden:    int   = 64,
        embed_dim:      int   = 64,
        gru_dim:        int   = 64,
        decoder_hidden: int   = 32,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_dim, sage_hidden, embed_dim, dropout)
        self.temporal = TemporalGRU(embed_dim, gru_dim, dropout)
        self.decoder  = LinkDecoder(gru_dim, decoder_hidden)

    def forward(
        self,
        node_feat_seq:   List[torch.Tensor],   # L × (N, F)
        edge_index_seq:  List[torch.Tensor],   # L × (2, E_t)
        pair_idx:        torch.Tensor,          # (P, 2)
    ) -> torch.Tensor:                          # (P,) logits
        """
        Parameters
        ----------
        node_feat_seq  : list of L node feature tensors, one per snapshot
        edge_index_seq : list of L edge_index tensors, one per snapshot
        pair_idx       : (P, 2) tensor of (i,j) pairs to score

        Returns
        -------
        (P,) raw logits — apply sigmoid externally for probabilities
        """
        embeddings = []
        for x, ei in zip(node_feat_seq, edge_index_seq):
            h = self.encoder(x, ei)   # (N, embed_dim)
            embeddings.append(h)

        seq = torch.stack(embeddings, dim=0)   # (L, N, embed_dim)
        h_t = self.temporal(seq)               # (N, gru_dim)
        return self.decoder(h_t, pair_idx)     # (P,) logits


# ─────────────────────────────────────────────────────────────────────────────
# Focal loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss for severely imbalanced edge classification.

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where p_t = sigmoid(logit) for positive class,
          p_t = 1 - sigmoid(logit) for negative class.

    Parameters
    ----------
    gamma      : focusing parameter.  Recommended values:
                   w=35  target  (120:1 imbalance) → γ=2
                   w=120 target  (478:1 imbalance) → γ=3
                   w=180 target  (402:1 imbalance) → γ=3
    pos_weight : scalar weight on the positive class before focal down-
                 weighting.  Set to None for automatic estimation from
                 the training batch (recommended).  If set explicitly,
                 use the inverse positive rate, e.g. 120 for 120:1.
    reduction  : 'mean' (default) or 'sum'

    Notes
    -----
    Implementation follows Lin et al. (2017) "Focal Loss for Dense Object
    Detection".  The numerically stable version operates on logits to avoid
    computing log(sigmoid(x)) directly.
    """

    def __init__(
        self,
        gamma:      float = 2.0,
        pos_weight: Optional[float] = None,
        reduction:  str = "mean",
    ):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight
        self.reduction  = reduction

    def forward(
        self,
        logits: torch.Tensor,   # (P,) raw logits
        labels: torch.Tensor,   # (P,) binary {0., 1.}
    ) -> torch.Tensor:
        # BCE loss per element (numerically stable, no reduction)
        if self.pos_weight is not None:
            pw = torch.tensor(
                self.pos_weight, device=logits.device, dtype=logits.dtype
            )
        else:
            # Auto-estimate from batch: inverse positive rate
            pos_rate = labels.mean().clamp(min=1e-4, max=1 - 1e-4)
            pw = (1.0 - pos_rate) / pos_rate

        bce = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pw, reduction="none"
        )

        # Focal weighting: (1 - p_t)^gamma
        prob    = torch.sigmoid(logits)
        p_t     = prob * labels + (1.0 - prob) * (1.0 - labels)
        focal_w = (1.0 - p_t).pow(self.gamma)

        loss = focal_w * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Pair sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_pairs(
    adj_target: np.ndarray,
    neg_ratio:  int = 10,
    rng:        Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample positive pairs (edges in adj_target) and a fixed multiple of
    negative pairs (non-edges) from the upper triangle.

    Parameters
    ----------
    adj_target : (N, N) binary adjacency of the target scale at time t+1
    neg_ratio  : number of negatives per positive (default 10)
    rng        : numpy random Generator for reproducibility

    Returns
    -------
    pair_idx : (P, 2) int array of (i, j) pairs  (i < j)
    labels   : (P,)  float32 array {0., 1.}

    Notes
    -----
    If the graph has zero positive edges (possible at very sparse snapshots),
    returns a random negative-only sample of size 500.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = adj_target.shape[0]
    triu_i, triu_j = np.triu_indices(N, k=1)

    # Positive pairs
    flat = adj_target[triu_i, triu_j]
    pos_mask = flat == 1
    neg_mask = flat == 0

    pos_i = triu_i[pos_mask]
    pos_j = triu_j[pos_mask]
    neg_i = triu_i[neg_mask]
    neg_j = triu_j[neg_mask]

    n_pos = len(pos_i)

    if n_pos == 0:
        # Degenerate snapshot: return random negatives
        n_sample = min(500, len(neg_i))
        sel = rng.choice(len(neg_i), size=n_sample, replace=False)
        pair_idx = np.stack([neg_i[sel], neg_j[sel]], axis=1)
        labels   = np.zeros(n_sample, dtype=np.float32)
        return pair_idx, labels

    # Sample negatives at neg_ratio:1
    n_neg = min(n_pos * neg_ratio, len(neg_i))
    sel_neg = rng.choice(len(neg_i), size=n_neg, replace=False)

    all_i = np.concatenate([pos_i, neg_i[sel_neg]])
    all_j = np.concatenate([pos_j, neg_j[sel_neg]])
    all_labels = np.concatenate([
        np.ones(n_pos,  dtype=np.float32),
        np.zeros(n_neg, dtype=np.float32),
    ])

    # Shuffle
    perm = rng.permutation(len(all_labels))
    pair_idx = np.stack([all_i[perm], all_j[perm]], axis=1)
    labels   = all_labels[perm]

    return pair_idx, labels


# ─────────────────────────────────────────────────────────────────────────────
# Parameter count utility
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
