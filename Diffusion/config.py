import torch

# Hyperparameters
IMG_SIZE = 32
BATCH_SIZE = 64
NUM_TIMESTEPS = 1000
EPOCHS = 50
LR = 1e-4

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")