import torch
import timm
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import theseus as th
import copy
import os


class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)


# State Encoder (ho)
class StateEncoder(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(StateEncoder, self).__init__()
        self.obs_encoder = timm.create_model(
            'resnet18.a1_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )

        # Freeze the observation encoder
        for param in self.obs_encoder.parameters():
            param.requires_grad = False

        #mean-std normalization

        # self.obs_transform = transforms.Compose([
        #     transforms.Resize((96, 96)),  # Ensure input is 96x96
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, obs, state):
        # obs, state = obs
        normalized_obs = (obs - self.img_mean) / self.img_std
        with torch.no_grad():
        # with self.obs_encoder.eval():
            self.obs_encoder.eval()
            img_emb = self.obs_encoder.forward_features(normalized_obs)
        img_emb = img_emb.view(img_emb.size(0), -1).detach().clone()
        # normalized_state = state/512 * 2 - 1
        state = torch.cat((img_emb, state), dim=-1)
        return self.network(state)

# Action Encoder (ha)
class ActionEncoder(nn.Module):
    def __init__(self, action_dim, latent_dim_action):
        super(ActionEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim_action)
        )

    def forward(self, action):
        return self.network(action)

# Fusing Encoder (hl)
class FusingEncoder(nn.Module):
    def __init__(self, latent_dim_state, latent_dim_action, posterior_dim):
        super(FusingEncoder, self).__init__()
        self.mu_network = nn.Sequential(
            nn.Linear(latent_dim_state + latent_dim_action, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 2*posterior_dim)
        )
        self.posterior_dim = posterior_dim

    def forward(self, zs, za):
        z_fused = torch.cat((zs, za), dim=-1)
        z = self.mu_network(z_fused)
        mu = z[:, :self.posterior_dim]
        logvar = z[:, self.posterior_dim:]
        return mu, logvar

# Reward Decoder (r)
class RewardDecoder(nn.Module):
    def __init__(self, latent_dim_state, latent_dim_action, posterior_dim):
        super(RewardDecoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim_state + latent_dim_action + posterior_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, zs, za, z):
        z_combined = torch.cat((zs, za, z), dim=-1)
        return self.network(z_combined)

# Dynamics Function (d)
class DynamicsFunction(nn.Module):
    def __init__(self, latent_dim_state, latent_dim_action, posterior_dim):
        super(DynamicsFunction, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim_state + latent_dim_action + posterior_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, latent_dim_state + posterior_dim)
        )
        self.latent_dim_state = latent_dim_state

    def forward(self, zs, za, z):
        z_combined = torch.cat((zs, za, z), dim=-1)
        z = self.network(z_combined)
        return z[:, :self.latent_dim_state], z[:, self.latent_dim_state:]

# Decoder for Action Reconstruction
class ActionDecoder(nn.Module):
    def __init__(self, posterior_dim, latent_dim_state, action_dim):
        super(ActionDecoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(posterior_dim + latent_dim_state, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, z, zs):
        z_combined = torch.cat((z, zs), dim=-1)
        return self.network(z_combined)



# Loss Function (ELBO)
def compute_elbo_loss(reconstructed_action, expert_action, z_mu, z_logvar, beta):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed_action, expert_action, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss




