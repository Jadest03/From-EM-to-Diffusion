import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# config
img_size = 32
batch_size = 64
num_timesteps = 1000
epochs = 50
lr = 1e-4
device = torch.device("mps")

def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i+1)
            plt.imshow(images[i])
            plt.axis('off')
            i += 1
    plt.show()

def pos_encoding(ts, out_dim, device=device):
    batch_size = len(ts)
    vs = torch.zeros(batch_size, out_dim, device=device)
    k = torch.arange(0, out_dim, 2, device=device, dtype=torch.float32) # 0, 2, 4 ...
    div_term = 10000 ** (k / out_dim) 
    
    for i in range(batch_size):
        v = torch.zeros(out_dim, device=device)
        v[0::2] = torch.sin(ts[i] / div_term) # 0, 2, 4 ...
        v[1::2] = torch.cos(ts[i] / div_term) # 1, 3, 5 ...
        vs[i] = v
    return vs

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )
        
    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v) # BroadCasting((N, C, H, W) + (N, C, 1 + 1))
        return y
    
class UNetCond(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        
        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)
            
    def forward(self, x, timesteps, labels=None):
        t = pos_encoding(timesteps, self.time_embed_dim)
        
        if labels is not None:
            t += self.label_emb(labels)
            
        x1 = self.down1(x, t)
        x = self.maxpool(x1)
        x2 = self.down2(x, t)
        x = self.maxpool(x2)
        
        x = self.bot1(x, t)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, t)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, t)
        x = self.out(x)
        return x
        
class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device) # [0.0001, 0.000119, ...]
        self.alphas = 1 - self.betas # [0.9999, 0.99988, ...]
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) # [a_0, a_0*a_1, ...]
        
    # Get x_t and noise from x_0
    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()
        t_idx = t - 1 # (N, 1)
        
        alpha_bar = self.alpha_bars[t_idx] # (N, 1)
        alpha_bar = alpha_bar.view(alpha_bar.size(0), 1, 1, 1) # (N, 1, 1, 1)
        
        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise
    
    # p(x_(t-1) | x_t ; theta) 
    def denoise(self, model, x, t, labels, gamma):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()
        
        t_idx = t - 1 # (N, 1)
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx - 1]
        
        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)
        
        # 이미지 생성 단계이므로 평가모드로 진행
        # 노이즈를 조금씩 깎아가며 진행
        # 분류기 신경망이 없는 가이던스 적용
        model.eval()
        with torch.no_grad():
            eps_cond = model(x, t, labels)
            eps_uncond = model(x, t)
            eps = eps_uncond + gamma * (eps_cond - eps_uncond)
        model.train()
        
        # t==1 일 때는 노이즈 x
        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0 # 추가로, 음수 인덱스 방지
        
        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std
    
    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)
    
    def sample(self, model, x_shape=(20, 1, 28, 28), labels=None):
        batch_size = x_shape[0] # N
        x = torch.randn(x_shape, device=self.device)
        
        if labels is None:
            labels = torch.randint(0, 10, (len(x),), device=self.device)
        
        # t = 0 까지 노이즈 제거
        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long) # (N, 1)
            x = self.denoise(model, x, t, labels, gamma=3.0) # (N, c, h, w)
        
        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images
    
if __name__=="__main__":
    preprocess = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    diffuser = Diffuser(num_timesteps, device=device)
    model = UNetCond(in_ch=3, num_labels=10)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    print(device)
    
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
                labels = None
            
            x_noisy, noise = diffuser.add_noise(x, t)
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
    
    # 가중치 저장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, 'cfg_ddpm_cifar10.pth')
    torch.save(model.state_dict(), save_path)
    print("모델 저장됨.")
    
    # 이미지 생성 
    target = 1 # 1 : 자동차
    target_labels = torch.full((20,), target, device=device, dtype=torch.long)
    images = diffuser.sample(model, x_shape=(20, 3, 32, 32), labels=target_labels)
    show_images(images)
