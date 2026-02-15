import torch

# Device Setup
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Dataset Hyper Parameters (CIFAR-10)
INPUT_DIM = 3      
IMAGE_SIZE = 32
HIDDEN_DIM = 128   
LATENT_DIM = 128   
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
DEVICE = get_device()