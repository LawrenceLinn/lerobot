import torch
import timm
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

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

        #mean-std normalization


        # self.obs_transform = transforms.Compose([
        #     transforms.Resize((96, 96)),  # Ensure input is 96x96
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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
        with torch.inference_mode():
            img_emb = self.obs_encoder.forward_features(normalized_obs)
        img_emb = img_emb.view(img_emb.size(0), -1)
        normalized_state = state/512 * 2 - 1
        state = torch.cat((img_emb, normalized_state), dim=-1)
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
            nn.Linear(512, 1)
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
            nn.Linear(256, action_dim)
        )

    def forward(self, z, zs):
        z_combined = torch.cat((z, zs), dim=-1)
        return self.network(z_combined)

# CVAE with Default Network Details
class CVAEWithDefaultNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim_state, latent_dim_action, posterior_dim):
        super(CVAEWithDefaultNetwork, self).__init__()
        self.state_encoder = StateEncoder(state_dim, latent_dim_state)
        self.action_encoder = ActionEncoder(action_dim, latent_dim_action)
        self.fusing_encoder = FusingEncoder(latent_dim_state, latent_dim_action, posterior_dim)
        self.dynamics_function = DynamicsFunction(latent_dim_state, latent_dim_action, posterior_dim)
        self.action_decoder = ActionDecoder(posterior_dim, latent_dim_state, action_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state, action):
        # Encode state and action
        zs = self.state_encoder(state)          # Encoded state
        za = self.action_encoder(action)        # Encoded action

        # Fuse state and action for posterior Gaussian
        z_mu, z_logvar = self.fusing_encoder(zs, za)

        # Sample from posterior Gaussian
        fused_latent = self.reparameterize(z_mu, z_logvar)

        # Reconstruct action using decoder
        reconstructed_action = self.action_decoder(fused_latent, zs)

        # Dynamics prediction (optional for future state prediction)
        _zs = self.dynamics_function(zs, za, fused_latent)

        return reconstructed_action, z_mu, z_logvar, _zs

# Loss Function (ELBO)
def elbo_loss(reconstructed_action, expert_action, z_mu, z_logvar, beta):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed_action, expert_action, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

    return recon_loss + beta * kl_loss


