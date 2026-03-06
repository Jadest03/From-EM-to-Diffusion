import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *
from unet import UNet
from ddpm import DDPM

def train():
    print(f"Device: {DEVICE}")
    
    preprocess = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    diffuser = DDPM(NUM_TIMESTEPS, device=DEVICE)
    model = UNet(in_ch=3).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    
    losses = []
    
    for epoch in range(EPOCHS):
        loss_sum = 0.0
        cnt = 0
        
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            x = images.to(DEVICE)
            t = torch.randint(1, NUM_TIMESTEPS+1, (len(x),), device=DEVICE)
            
            x_noisy, noise = diffuser.add_noise(x, t)
            noise_pred = model(x_noisy, t)
            
            loss = F.mse_loss(noise, noise_pred)
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            cnt += 1
            
        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f'Epoch {epoch+1} | Loss : {loss_avg:.4f}')
        
    # save model
    final_save_path = 'ddpm_final.pth'
    torch.save(model.state_dict(), final_save_path)
    print("model saved")
    
    # Loss 그래프 출력
    plt.plot(losses)
    plt.title("Training Loss")
    plt.show()

if __name__=="__main__":
    train()