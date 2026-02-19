import torch
import torch.nn as nn
import torch.nn.functional as F

# Reparameterization Trick
def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        
        # flatten한 벡터 차원 수 x 높이 x 너비
        self.final_feature_dim = (hidden_dim * 4) * 4 * 4
        
        # 평균 벡터, 로그 분산
        self.linear_mu = nn.Linear(self.final_feature_dim, latent_dim)
        self.linear_logvar = nn.Linear(self.final_feature_dim, latent_dim)
        
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.flatten(h)
        
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.final_feature_dim = (hidden_dim * 4) * 4 * 4
        
        self.linear = nn.Linear(latent_dim, self.final_feature_dim)

        self.deconv1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, z):
        h = self.linear(z)
        h = h.view(-1, self.hidden_dim * 4, 4, 4)
        
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        x_hat = torch.sigmoid(self.deconv3(h))
        return x_hat

# VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        batch_size = len(x)
        
        # MSE Loss(Gaussian 이므로)
        reconstruction = F.mse_loss(x_hat, x, reduction='sum')
        # KL Loss
        KL = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
        
        return (reconstruction + KL) / batch_size