import os
import torch
import torchvision
import matplotlib.pyplot as plt
import config
from vae import VAE
device = config.DEVICE

# 모델 준비
model = VAE(config.INPUT_DIM, config.HIDDEN_DIM, config.LATENT_DIM).to(device)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'vae_cifar.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("모델을 가져왔습니다.")
except:
    print("모델 파일이 없습니다.")

# Generation
model.eval()
with torch.no_grad():
    sample_size = 64
    z = torch.randn(sample_size, config.LATENT_DIM).to(device)
    
    # 학습된 decoder 모델 가져오기
    x = model.decoder(z)
    
    # CIFAR-10 -> 3 channel
    generated_imges = x.view(sample_size, 3, 32, 32).cpu()

# 이미지 한 번에 출력
grid_img = torchvision.utils.make_grid(
    generated_imges,
    nrow=8,
    padding=2,
    normalize=True
)

plt.imshow(grid_img.permute(1, 2, 0))
plt.axis('off')
plt.show()