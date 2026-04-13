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
