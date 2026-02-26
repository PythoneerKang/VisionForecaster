#Window widths
w = 35

#Git/GitHub Repo name
repo_name = "VisionForecaster"

#Save dataloaders?
save_loaders = False

#Number of training epochs
num_epochs = 50

# Hyperparameters
IMG_SIZE = 457
PATCH_SIZE = 16
CHANNELS = 1
EMBED_DIM = 64 #256
HEADS = 8
DEPTH = 3 #6
MLP_DIM = 4 * EMBED_DIM #Often 4 times EMBED_DIM #D_FF = 4 * D_MODEL , ior 512
BATCH_SIZE = 128

#V100 -- cuda 12.7
#A40 -- cuda 12.7
#H100 -- unknown