import torch
import torchvision.transforms as transforms
from tqdm import tqdm
     
class CFG:
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
            # null token에 값 대입
            null_labels = torch.full_like(labels, 10) 
            eps_uncond = model(x, t, null_labels)
            eps = eps_uncond + gamma * (eps_cond - eps_uncond)
        model.train()
        
        # when t==1, noise x
        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0 # prevent minus index
        
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
    
    def sample(self, model, x_shape=(20, 1, 28, 28), labels=None, gamma=3.0):
        batch_size = x_shape[0] # N
        x = torch.randn(x_shape, device=self.device)
        
        if labels is None:
            labels = torch.randint(0, 10, (len(x),), device=self.device)
        
        # denoise until t = 0 
        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long) # (N, 1)
            x = self.denoise(model, x, t, labels, gamma=gamma) # (N, c, h, w)
        
        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images
    


