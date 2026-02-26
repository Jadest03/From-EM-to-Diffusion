import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
img_size = 32
batch_size = 64
num_timesteps = 1000
epochs = 300
lr = 1e-4
gamma = 6.0
device = get_device()