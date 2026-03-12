# =============================================================================
# parameters.py — central configuration for SmallDataDecoderViT training
# =============================================================================

# Window width for correlation computation.
# Confirmed with supervisor (Dr. Cheong): use w=35 days only.
w = 35

# Git/GitHub Repo name
repo_name = "VisionForecaster"

# Number of training epochs (per fold)
num_epochs = 100

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

# BATCH_SIZE: with max 504 training samples per fold and BATCH_SIZE=8,
# each epoch produces ~63 gradient steps — a healthy number for convergence.
# Recommended range: 4–16. Increase cautiously if training is stable.
BATCH_SIZE = 8

# -----------------------------------------------------------------------------
# Legacy hyperparameters — NOT used by SmallDataDecoderViT
# These were used by the previous VisionForecaster model and are kept here
# only for reference. Do not use these to configure the current model.
# The active model config lives in diff_model_multi_fold_cv_train_test()
# in training_and_validation_functions.py.
# -----------------------------------------------------------------------------
# IMG_SIZE  = 457
# PATCH_SIZE = 16
# CHANNELS  = 1
# EMBED_DIM = 64
# HEADS     = 8
# DEPTH     = 3
# MLP_DIM   = 4 * EMBED_DIM
# BATCH_SIZE (legacy) = 128

# -----------------------------------------------------------------------------
# GPU notes (for future reference)
# -----------------------------------------------------------------------------
# V100 -- cuda 12.7
# A40  -- cuda 12.7
# H100 -- unknown
