import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import UNet
from scheduler import linear_beta_schedule, cosine_beta_schedule
from diffuse import forward_diffusion, sample

from utils import show_images

# Hyperparameters
timesteps = 1000
in_channels = 3  # CIFAR-10 images have 3 color channels
image_size = 64
batch_size = 64
epochs = 20000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = datasets.CIFAR10(root='cifar_data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# Beta schedule and other constants
betas = cosine_beta_schedule(timesteps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# Model, optimizer, and training
model = UNet(in_channels=in_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        t = torch.randint(0, timesteps, (images.size(0),), device=device).long()
        noisy_image, noise = forward_diffusion(images, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        predicted_noise = model(noisy_image, t)

        loss = criterion(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    if (epoch + 1) % 5 == 0:
        sample_images = sample(model, image_size=64, batch_size=8, timesteps=timesteps, betas=betas, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, device=device)
        show_images(sample_images)

