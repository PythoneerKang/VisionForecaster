# =============================================================================
# parameters.py — central configuration for SmallDataDecoderViT training
# =============================================================================

# Window width for correlation computation.
# Confirmed with supervisor (Dr. Cheong): use w=35 days only.
w = 35

# TDA Filtration threshold
epsilon = 0.1

# Git/GitHub Repo name
repo_name = "VisionForecaster"

# Number of training epochs (per fold)
num_epochs = 100

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Hardware / runtime configuration for HPC CPU training
# -----------------------------------------------------------------------------

# Target number of physical/logical CPU cores on the node.
# PBS script requests 19 cores — keep this consistent with the script.
NUM_CPUS = 19

# Torch threading configuration.
#   TORCH_NUM_THREADS      : intra-op parallelism (BLAS / math work per op)
#   TORCH_NUM_INTEROP_THREADS : parallelism across independent ops
# Rule of thumb: NUM_WORKERS * TORCH_NUM_THREADS + TORCH_NUM_INTEROP_THREADS
#   should be <= NUM_CPUS.  2*8 + 2 = 18 < 19. Fine.
# To better saturate 19 cores you could raise TORCH_NUM_THREADS to 10-12
# and reduce NUM_WORKERS to 1.
TORCH_NUM_THREADS = 8
TORCH_NUM_INTEROP_THREADS = 2

# DataLoader worker processes.
NUM_WORKERS = 2

# Whether to use GPU when available.
# Keep False for Intel CPU-only HPC nodes.
USE_GPU = False

# -----------------------------------------------------------------------------
# Model & training hyperparameters
# -----------------------------------------------------------------------------

# BATCH_SIZE: with max 504 training samples per fold and BATCH_SIZE=16,
# each epoch produces ~31 gradient steps.  shuffle=True in the DataLoader
# means these are 31 diverse steps — consecutive batches are not from the
# same market-regime window.
BATCH_SIZE = 16

# -----------------------------------------------------------------------------
# GPU notes (for future reference)
# -----------------------------------------------------------------------------
# V100 -- cuda 12.7
# A40  -- cuda 12.7
# H100 -- unknown


# =============================================================================
# GNN cross-scale prediction hyperparameters
# (used by gnn_cross_scale.py and train_cross_scale.py)
# =============================================================================

# Node embedding dimension shared by GraphSAGE encoder and GRU hidden state.
# 64 gives a good capacity / overfitting trade-off at ~220 weekly training steps.
# Reduce to 32 for a quick smoke test; increase to 128 only with GPU.
GNN_EMBED_DIM = 64

# Number of weekly history snapshots fed to the GRU.
# 4 steps = 20 trading days of short-scale memory.
# Increasing beyond 8 rarely helps given the regime stability of A_w120/A_w180.
GNN_HISTORY_LAGS = 4

# Negative:positive pair sampling ratio during training.
# 10:1 keeps the training distribution balanced while covering diverse negatives.
# Reduce to 5 if training is slow on CPU; increase to 20 if precision is low.
GNN_NEG_RATIO = 5

# Focal loss gamma per target scale.
# Higher gamma → stronger down-weighting of easy negatives.
#   A_w120 target (imbalance ~478:1) → γ=1
#   A_w180 target (imbalance ~402:1) → γ=2
# This will shift the precision-recall balance rightward — recall will drop from ~75%
# to perhaps ~50%, but precision will improve from ~10% to ~30–40%, and AP should
# increase substantially.
# These are stored in TARGET_CONFIGS inside train_cross_scale.py and
# check_cross_scale_learnability.py; kept here for reference / override.
GNN_FOCAL_GAMMA_W120 = 1
GNN_FOCAL_GAMMA_W180 = 2

# Dropout rate applied inside GraphSAGE layers and after GRU output.
GNN_DROPOUT = 0.1

# Training epochs per fold.
# 100 is the default; early stopping (patience=15) will typically fire
# at 30–60 epochs once the model has converged.
GNN_EPOCHS = 100

# AdamW hyperparameters for the GNN optimizer.
GNN_LR           = 3e-4
GNN_WEIGHT_DECAY = 1e-3

# Holdout fraction for the expanding-window evaluation split.
# 1/6 ≈ 0.167 gives ~52 holdout steps with 315 weekly snapshots,
# matching the last fold of the previous 5-fold TimeSeriesSplit scheme.
GNN_HOLDOUT_FRAC = 1.0 / 6.0
