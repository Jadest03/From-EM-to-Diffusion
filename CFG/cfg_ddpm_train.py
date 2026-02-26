import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import config
import unet
import cfg_ddpm

if __name__ == "__main__":
    # load hyperparameter
    img_size = config.img_size
    batch_size = config.batch_size
    num_timesteps = config.num_timesteps
    epochs = config.epochs
    lr = config.lr
    GAMMA = config.gamma
    device = config.device

    # define model
    preprocess = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cfg = cfg_ddpm.CFG(num_timesteps, device=device)
    model = unet.UNetCond(in_ch=3, num_labels=11)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    print(device)
    
    # train
    losses = []
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        
        for images, labels in tqdm(dataloader):
            optimizer.zero_grad()
            x = images.to(device)
            labels = labels.to(device)
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device) # (N, )
            
            if np.random.random() < 0.1:
                labels = torch.full_like(labels, 10)
            
            x_noisy, noise = cfg.add_noise(x, t)
            noise_pred = model(x_noisy, t, labels)
            loss = F.mse_loss(noise, noise_pred)
            
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            cnt += 1
            
        # loss per 1 epoch
        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f'Epoch[{epoch + 1}/{epochs}] | Loss : {loss_avg}')

    plt.plot(losses)
    plt.show()

    # save model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, 'cfg_ddpm_cifar10.pth')
    torch.save(model.state_dict(), save_path)
    print("model saved")