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
        return F.tanh(self.network(z_combined))

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

    return recon_loss + beta * kl_loss



import torch
import torch.nn as nn
import torch.nn.functional as F
import theseus as th
import copy
from cvae_utilities import *
import copy
import os
import wandb
from torch.cuda.amp import autocast, GradScaler


class CVAEWithTrajectoryOptimization(nn.Module):
    def __init__(self, cfg):
        super(CVAEWithTrajectoryOptimization, self).__init__()
        state_dim = cfg.state_dim
        action_dim = cfg.action_dim * cfg.horizon
        latent_dim_state = cfg.latent_dim_state 
        latent_dim_action = cfg.latent_dim_action * cfg.horizon
        posterior_dim = cfg.posterior_dim
        device = cfg.device
        self.state_encoder = StateEncoder(state_dim, latent_dim_state).to(device)
        self.action_encoder = ActionEncoder(action_dim, latent_dim_action).to(device)
        self.fusing_encoder = FusingEncoder(latent_dim_state, latent_dim_action, posterior_dim).to(device)
        self.dynamics_function = DynamicsFunction(latent_dim_state, latent_dim_action, posterior_dim).to(device)
        self.action_decoder = ActionDecoder(posterior_dim, latent_dim_state, action_dim).to(device)
        self.reward_decoder = RewardDecoder(latent_dim_state, latent_dim_action, posterior_dim).to(device)
        self.posterior_dim = posterior_dim
        self.device = device
        self.cfg = cfg

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, zs, za):
        """
        Perform a forward pass through the model.
        Args:
            obs: Observation (raw img input), shape [batch_size, state_dim].
            state: State (latent state input), shape [batch_size, state_dim].
            action: Action (latent action input), shape [batch_size, horizon * action_dim].
        Returns:
            reconstructed_action: Reconstructed action, shape [batch_size, horizon, action_dim].
            z_mu: Mean of the posterior Gaussian, shape [batch_size, posterior_dim].
            z_logvar: Log-variance of the posterior Gaussian, shape [batch_size, posterior_dim].
            _zs: Predicted next latent state, shape [batch_size, state_dim].
        """
    
        # Encode state and action
        # zs = self.state_encoder(obs, state)
        # za = self.action_encoder(action)

        # Fuse state and action for posterior Gaussian
        z_mu, z_logvar = self.fusing_encoder(zs, za)

        # Sample from posterior Gaussian
        fused_latent = self.reparameterize(z_mu, z_logvar)

        reconstructed_action = self.action_decoder(fused_latent, zs)

        # Predict next latent state
        _zs = self.dynamics_function(zs, self.action_encoder(reconstructed_action), fused_latent)#TODO: check if this is correct

        return reconstructed_action, z_mu, z_logvar, _zs
    
    def generate_action(self, zs, zp):
        """
        Generate an action based on the given observation.
        Args:
            obs: Observation (raw state input), shape [batch_size, state_dim].
            state: State (latent state input), shape [batch_size, state_dim].
            z: Latent variable z, shape [batch_size, posterior_dim].
        Returns:
            action: Generated action, shape [batch_size, action_dim].
        """

        # Decode action from latent state and z
        reconstructed_action = self.action_decoder(zp, zs)  # Reconstruct action

        _zs, _zp = self.dynamics_function(zs, self.action_encoder(reconstructed_action), zp)

        return reconstructed_action, _zs, _zp

    def save_pretrained(self, save_directory):
        """Save model weights and config to directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "model.pt")
        torch.save({
            'state_encoder': self.state_encoder.state_dict(),
            'action_encoder': self.action_encoder.state_dict(),
            'fusing_encoder': self.fusing_encoder.state_dict(),
            'dynamics_function': self.dynamics_function.state_dict(),
            'action_decoder': self.action_decoder.state_dict(),
            'reward_decoder': self.reward_decoder.state_dict(),
            # 'cfg': self.cfg
        }, model_path)

    def load_pretrained(self, load_directory):
        """Load model weights and config from directory"""
        model_path = os.path.join(load_directory, "model.pt")
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load model weights
        self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.action_encoder.load_state_dict(checkpoint['action_encoder'])
        self.fusing_encoder.load_state_dict(checkpoint['fusing_encoder'])
        self.dynamics_function.load_state_dict(checkpoint['dynamics_function'])
        self.action_decoder.load_state_dict(checkpoint['action_decoder'])
        self.reward_decoder.load_state_dict(checkpoint['reward_decoder'])
        
        # Load config
        # self.cfg = checkpoint['cfg']

    def plan_with_theseus_update(self, obs, state, action, horizon, gamma, cfg, eval_mode=False):
        """
        Perform trajectory optimization using Theseus.
        Args:
            obs: Observation (raw state input).
            horizon: Planning horizon.
            gamma: Discount factor.
            model: The CVAE model containing dynamics and state encoders.
            cfg: Configuration object (for damping, step size, etc.).
            eval_mode: Whether to evaluate without gradients.
        """
        # Prepare initial observation
        batch_size = obs.shape[0]
        obs = torch.tensor(obs[:,0,...], dtype=torch.float32, device=self.device) # [bs, 3, 96, 96]
        state = torch.tensor(state[:,0,...], dtype=torch.float32, device=self.device)# [bs, 2]
        action = torch.tensor(action, dtype=torch.float32, device=self.device) # [bs, horizon * 2 * 2]
        horizon = int(min(horizon, cfg.horizon))  # Clip planning horizon

        # Initialize latent state and actions
          # Latent state
        # actions = torch.zeros(batch_size, horizon, cfg.action_dim, device=self.device, requires_grad=True)  # Initial actions

        # Precompute initial actions using policy (if available)
        pi_actions = torch.empty(batch_size, horizon, cfg.action_dim * horizon, device=self.device)
        expert_actions = torch.empty(batch_size, horizon, cfg.action_dim * horizon, device=self.device)
        z_mus = []
        z_logvars = []

        zs = self.state_encoder(obs, state) # 【bs, 50]
        zs0 = zs.clone()
        # pz = torch.randn(batch_size, model.posterior_dim, device=self.device)
        # pz0 = pz.clone()
        # with torch.no_grad():
        # consistance_loss = 0 #TODO: calculate consistance loss by mse(_zs) KL(zp|N(0,1))

        zp = torch.randn(batch_size, self.posterior_dim, device=self.device)

        for t in range(horizon):
            # pi_actions[:,t] = model.generate_action(obs, state, pz)
            if not eval_mode:
                za = self.action_encoder(action[:,2*t:2*(horizon+t)]) # [bs, 14]
                reconstructed_action, z_mu, z_logvar, (zs0, zp) = self.forward(zs0, za) # [bs, 14]
                z_mus.append(z_mu)
                z_logvars.append(z_logvar)
            else:
                with torch.no_grad():
                    reconstructed_action, zs0, zp = self.generate_action(zs0, zp)
            pi_actions[:,t] = reconstructed_action
            expert_actions[:,t] = action[:,2*t:2*(horizon+t)]
        
        
        # Define cost function
        def value_cost_fn(optim_vars, aux_vars):
            actions = optim_vars[0].tensor  # [bs, horizon * action_dim]

            obs = aux_vars[0].tensor
            state = aux_vars[1].tensor
            pz = aux_vars[2].tensor

            actions = actions.view(batch_size, horizon, cfg.action_dim * horizon)
            obs = obs.squeeze(0)
            state = state.squeeze(0)
            pz = pz.squeeze(0)

            z = self.state_encoder(obs, state)  # Latent state
            total_reward = 0.0
            discount = 1.0
            # pz = torch.randn(batch_size, self.posterior_dim, device=self.device)
            # Compute cumulative reward
            for t in range(horizon):
                reward = self.reward_decoder(z, actions[:,t], pz)
                z, pz = self.dynamics_function(z, actions[:,t], pz)
                total_reward += discount * reward
                discount *= gamma
            err = -torch.nan_to_num(total_reward, nan=0.0) + 1e3
            return err.view(1,-1)
        
        init_actions = pi_actions.view(1, -1)  
        input_obs = obs.unsqueeze(0)
        input_state = state.unsqueeze(0)
        input_pz = zp.unsqueeze(0)

        actions_var = th.Vector(tensor = init_actions, name="actions")
        obs_var = th.Variable(input_obs, name="obs")
        state_var = th.Variable(input_state, name="state")
        posterior = th.Variable(input_pz, name="posterior")

        cost_function = th.AutoDiffCostFunction(
            [actions_var],
            value_cost_fn,
            dim=batch_size,
            aux_vars=[obs_var, state_var, posterior],
            name="value_cost_fn",
        )

        objective = th.Objective()
        objective.add(cost_function)
        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=cfg.max_iterations,
            step_size=cfg.step_size,
        )
        theseus_layer = th.TheseusLayer(optimizer)
        theseus_layer.to(device=self.device)

        theseus_inputs = {"actions": init_actions, "obs": input_obs, "state": input_state, "posterior": input_pz}
        
        # Solve optimization problem
        updated_inputs, info = theseus_layer.forward(
            theseus_inputs,
            optimizer_kwargs={
                "track_best_solution": True,
                "damping": cfg.damping,
                "verbose": False,
                "backward_mode" : "truncated",
                "backward_num_iterations":10,
            },
        )
        best_actions = info.best_solution["actions"].view(batch_size, horizon, horizon * cfg.action_dim)
        updated_actions = updated_inputs['actions'].nan_to_num_(0).view(batch_size, horizon, horizon * cfg.action_dim)

        if not eval_mode:
            # bc_loss = F.mse_loss(updated_actions[:,:1,:], expert_actions[:,:1,:])
            z_mus = torch.stack(z_mus, dim=1)
            z_logvars = torch.stack(z_logvars, dim=1)

            elbo_loss = compute_elbo_loss(updated_actions[:,:,:6], expert_actions[:,:,:6], z_mus, z_logvars, cfg.beta)
            output_dict = {
                "best_actions": best_actions,
                "elbo_loss": elbo_loss,
            }
        else:
            bc_loss = F.mse_loss(best_actions[:,:,:6], expert_actions[:,:,:6].cpu())
            output_dict = {
                "bc_loss": bc_loss,
                "best_actions": best_actions,
            }
        return output_dict
    
    




def generate_reward_map(policy, obs, state, resolution=100):
    """Generate a reward landscape by scanning over possible actions.
    Args:
        policy: The trained CVAE policy
        obs: Image observation tensor of shape (1,3,96,96) 
        state: State tensor of shape (1,2)
        resolution: Number of points to sample in each action dimension
        
    Returns:
        actions: (resolution*resolution, 14) tensor of sampled actions
        rewards: (resolution*resolution,) tensor of predicted rewards
    """
    # Encode state and observation
    with torch.no_grad():
        zs = policy.state_encoder(obs, state)
        
        # Create action grid from (-1,-1) to (1,1)
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Create full action tensor with zeros
        actions = torch.zeros((resolution*resolution, 14), device=obs.device)
        # Fill first two dimensions with grid values
        actions[:, 0] = xx.flatten()
        actions[:, 1] = yy.flatten()
        
        # Encode each action
        za_list = []
        rewards = []

        # Process in batches to avoid memory issues
        batch_size = 100
        z = torch.randn(batch_size,64).cuda()
        for i in range(0, len(actions), batch_size):
            action_batch = actions[i:i+batch_size]
            
            # Encode action
            za = policy.action_encoder(action_batch)
            
            # # Get latent from encoder
            # z_mu, z_logvar = policy.fusing_encoder(
            #     zs.repeat(len(action_batch),1), 
            #     za
            # )

            # z = policy.reparameterize(z_mu,z_logvar)
            
            
            # Decode reward
            reward = policy.reward_decoder(
                zs.repeat(len(action_batch),1),
                za, 
                z
            )
            rewards.append(reward)
            
        rewards = torch.cat(rewards, dim=0)
        
        rewards = rewards.squeeze(-1).cpu().numpy()

        action = actions.cpu().numpy()

        # Reshape rewards for plotting
        reward_grid = rewards.reshape(100,100)

        # Plot with matplotlib
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.imshow(reward_grid, extent=[-1,1,-1,1], origin='lower')
        plt.colorbar(label='Predicted Reward')
        plt.xlabel('Action Dim 1')
        plt.ylabel('Action Dim 2') 
        plt.title('Reward Landscape')
        plt.show()

        return 

# Example usage:
# obs = batch['observation.image'][0].cuda()
# state = batch['observation.state'][0].cuda()/512 * 2 - 1
# generate_reward_map(policy, obs, state)


