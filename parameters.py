#Window widths
w = 120

#Git/GitHub Repo name
repo_name = "VisionForecaster"

#Save dataloaders?
save_loaders = False

#Number of training epochs
num_epochs = 100

# -----------------------------------------------------------------------------
# Hardware / runtime configuration for HPC CPU training
# -----------------------------------------------------------------------------

# Target number of physical/logical CPU cores to use on the node.
# Your PBS script will request 16 cores; keep this in [10, 16].
NUM_CPUS = 16

# Torch threading configuration. These are used in the training code to
# control intra-op and inter-op parallelism on CPU.
# With 16 CPUs, a good starting point is 8 intra-op threads and 2 inter-op
# threads, plus a couple of DataLoader workers, to avoid oversubscription.
TORCH_NUM_THREADS = 8           # math / BLAS work per operator
TORCH_NUM_INTEROP_THREADS = 2   # parallelism across operators

# DataLoader worker processes. Keep roughly
#   NUM_WORKERS * TORCH_NUM_THREADS <= NUM_CPUS.
NUM_WORKERS = 2

# Whether to use GPU when available. For Intel CPU-only training on HPC, keep
# this False so training always runs on CPU even if a GPU is visible.
USE_GPU = False

# Hyperparameters
IMG_SIZE = 457
PATCH_SIZE = 16
CHANNELS = 1
EMBED_DIM = 64 #256
HEADS = 8
DEPTH = 3 #6
MLP_DIM = 4 * EMBED_DIM #Often 4 times EMBED_DIM #D_FF = 4 * D_MODEL , or 512
BATCH_SIZE = 128

#V100 -- cuda 12.7
#A40 -- cuda 12.7
#H100 -- unknown