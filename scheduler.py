import torch

# Beta schedule for the forward process: a linear noise schedule
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, timesteps)
    # Clamp betas to ensure stability
    betas = torch.clamp(betas, min=1e-5, max=0.999)
    return betas

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for betas as proposed in improved DDPM paper.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(betas, dtype=torch.float32)