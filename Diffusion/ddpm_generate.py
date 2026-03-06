import os
import torch
import matplotlib.pyplot as plt
from config import *
from unet import UNet
from ddpm import DDPM

def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i+1)
            plt.imshow(images[i])
            plt.axis('off')
            i += 1
    plt.tight_layout()
    plt.show()

def generate():
    model = UNet(in_ch=3).to(DEVICE)
    ddpm = DDPM(NUM_TIMESTEPS, device=DEVICE)
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'ddpm_final.pth')
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"모델 파일이 없습니다.")
    
    # Sampling
    print("Generating images...")
    images = ddpm.sample(model, x_shape=(20, 3, IMG_SIZE, IMG_SIZE))
    show_images(images)

if __name__=="__main__":
    generate()