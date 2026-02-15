import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import config
import dataset
from vae import VAE

# 설정값 가져오기
input_dim = config.INPUT_DIM
hidden_dim = config.HIDDEN_DIM
latent_dim = config.LATENT_DIM
epochs = config.EPOCHS
learning_rate = config.LEARNING_RATE
batch_size = config.BATCH_SIZE
device = config.DEVICE

print(f"Using device: {device}")

# 데이터 로더
dataloader = dataset.get_dataloader(batch_size)

# 모델 초기화
model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

print("Training started...")
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0
    for x, label in dataloader:
        x = x.to(device)
        
        optimizer.zero_grad()
        loss = model.get_loss(x)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        cnt += 1
        
    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"Epoch {epoch+1} loss: {loss_avg}")

# 현재 폴더에 모델 저장
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, 'vae_cifar.pth')
torch.save(model.state_dict(), save_path)
print("Model Saved!")

# 그래프
epoch_list = list(range(1, epochs + 1))
plt.plot(epoch_list, losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()