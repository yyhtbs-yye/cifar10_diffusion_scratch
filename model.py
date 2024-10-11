import torch
import torch.nn as nn
import math
# Sinusoidal position encoding for timestep conditioning (used in the U-Net)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# U-Net with timestep conditioning for denoising
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=None, time_dim=256):
        super(UNet, self).__init__()

        if out_channels is None: 
            out_channels = in_channels
        # Positional embedding for the time step
        self.time_embedding = SinusoidalPositionEmbeddings(time_dim)
        
        # Encoder
        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.encoder3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

        # Time embedding linear layers
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU())

        # Decoder
        self.decoder3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2))
        self.decoder2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2))
        self.decoder1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        # Final output layer
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Encode
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Combine with time embedding
        x3 = x3 + t_emb[:, :, None, None]

        # Bottleneck
        x4 = self.bottleneck(x3)

        # Decode
        x5 = self.decoder3(x4)
        x6 = self.decoder2(x5)
        x7 = self.decoder1(x6)

        return self.final(x7)
