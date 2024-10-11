import torch
import math

# Forward diffusion process: Add noise to the image based on the timestep
def forward_diffusion(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x_0)
    sqrt_alpha_t = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    noisy_image = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
    return noisy_image, noise

# Sampling process using reverse diffusion (denoising)
@torch.no_grad()
def sample(model, in_channels, image_size, batch_size, timesteps, betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device):
    model.eval()
    img = torch.randn((batch_size, in_channels, image_size, image_size)).to(device)
    
    for t in reversed(range(1, timesteps)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        predicted_noise = model(img, t_tensor)

        sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) + 1e-8  # Add epsilon for numerical stability

        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = betas[t].view(-1, 1, 1, 1)

        # Subtract the predicted noise component scaled by beta_t and sqrt(1 - alpha_cumprod)
        img = (img - (1 - sqrt_alpha_t) / sqrt_one_minus_alpha_t * predicted_noise) / torch.sqrt(sqrt_alpha_t)

        img = torch.clamp(img, -1.0, 1.0)  # Clipping values to a reasonable range

        # Add Gaussian noise at the current step, scaled by the noise factor
        if t > 1:
            noise = torch.randn_like(img)
            img = img + torch.sqrt(beta_t) * noise
    
    return img

