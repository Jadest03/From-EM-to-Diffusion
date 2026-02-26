import os
import torch
import matplotlib.pyplot as plt
import config
import unet
import cfg_ddpm

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

if __name__ == "__main__":
    device = config.device
    num_timesteps = config.num_timesteps
    gamma = config.gamma
    
    cfg = cfg_ddpm.CFG(num_timesteps, device=device)
    model = unet.UNetCond(in_ch=3, num_labels=11)
    model.to(device)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(current_dir, 'cfg_ddpm_cifar10.pth')
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print("model loaded")
    
    model.eval()
    target = 1 # 1 : car
    n = 10
    target_labels = torch.full((n,), target, device=device, dtype=torch.long)
    print("generating images...")
    images = cfg.sample(model, x_shape=(n, 3, 32, 32), labels=target_labels, gamma=gamma)
    show_images(images, 2, 5)