import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified RNNBase class
class RNNBase(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, rnn_type="LSTM", output_dim=None):
        super(RNNBase, self).__init__()
        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim) if output_dim else None

    def forward(self, x, rnn_init_state=None):
        # x: [batch_size, seq_len, input_dim]
        rnn_out, rnn_state = self.rnn(x, rnn_init_state)
        if self.output_layer:
            output = self.output_layer(rnn_out)  # [batch_size, seq_len, output_dim]
        else:
            output = rnn_out  # [batch_size, seq_len, hidden_dim]
        return output, rnn_state

    def forward_step(self, x, rnn_state):
        # x: [batch_size, input_dim]
        x = x.unsqueeze(1)  # Add seq_len dimension
        rnn_out, rnn_state = self.rnn(x, rnn_state)
        if self.output_layer:
            output = self.output_layer(rnn_out[:, -1, :])  # [batch_size, output_dim]
        else:
            output = rnn_out[:, -1, :]  # [batch_size, hidden_dim]
        return output, rnn_state

# State Encoder (ho)
class StateEncoder(nn.Module):
    def __init__(self, state_dim, latent_dim_state):
        super(StateEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim_state)
        )

    def forward(self, state):
        return self.network(state)  # Encoded latent state (zs)

# Action Encoder (ha) with simplified RNNBase
class ActionEncoderRNN(nn.Module):
    def __init__(self, action_dim, latent_dim_action, rnn_hidden_dim, rnn_num_layers, rnn_type="LSTM"):
        super(ActionEncoderRNN, self).__init__()
        self.rnn = RNNBase(
            input_dim=action_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            output_dim=latent_dim_action
        )

    def forward(self, action_seq):
        # action_seq: [batch_size, seq_len, action_dim]
        latent_action_seq, _ = self.rnn(action_seq)
        # Take the output at the last time step
        latent_action = latent_action_seq[:, -1, :]  # [batch_size, latent_dim_action]
        return latent_action

# Fusing Encoder (hl)
class FusingEncoder(nn.Module):
    def __init__(self, latent_dim_state, latent_dim_action, posterior_dim):
        super(FusingEncoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim_state + latent_dim_action, 256)
        self.fc_mu = nn.Linear(256, posterior_dim)      # Mean of posterior
        self.fc_logvar = nn.Linear(256, posterior_dim)  # Log variance of posterior

    def forward(self, zs, za):
        z_fused = torch.cat((zs, za), dim=-1)  # Combine latent state and action
        x = F.elu(self.fc1(z_fused))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Decoder with GMM
class DecoderWithGMM(nn.Module):
    def __init__(self, posterior_dim, latent_dim_state, gmm_num_modes, action_dim):
        super(DecoderWithGMM, self).__init__()
        self.gmm_num_modes = gmm_num_modes
        self.fc1 = nn.Linear(posterior_dim + latent_dim_state, 512)
        self.fc2_pi = nn.Linear(512, gmm_num_modes)           # Mixture weights
        self.fc2_mu = nn.Linear(512, gmm_num_modes * action_dim)     # Means
        self.fc2_sigma = nn.Linear(512, gmm_num_modes * action_dim)  # Variances

    def forward(self, z, zs):
        z_combined = torch.cat((z, zs), dim=-1)  # Combine z and zs
        x = F.elu(self.fc1(z_combined))
        pi = F.softmax(self.fc2_pi(x), dim=-1)   # Mixture weights
        mu = self.fc2_mu(x).view(-1, self.gmm_num_modes, -1)     # Means
        sigma = F.softplus(self.fc2_sigma(x)).view(-1, self.gmm_num_modes, -1)  # Variances (positive)
        return pi, mu, sigma

# CVAE Model with simplified RNNBase
class CVAEWithRNN(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim_state, latent_dim_action, posterior_dim, rnn_hidden_dim, rnn_num_layers, gmm_num_modes, rnn_type="LSTM"):
        super(CVAEWithRNN, self).__init__()
        self.state_encoder = StateEncoder(state_dim, latent_dim_state)
        self.action_encoder = ActionEncoderRNN(action_dim, latent_dim_action, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.fusing_encoder = FusingEncoder(latent_dim_state, latent_dim_action, posterior_dim)
        self.decoder = DecoderWithGMM(posterior_dim, latent_dim_state, gmm_num_modes, action_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state, action_seq):
        zs = self.state_encoder(state)          # Encode state to latent space
        za = self.action_encoder(action_seq)    # Encode action sequence to latent space
        mu, logvar = self.fusing_encoder(zs, za)  # Compute posterior parameters
        z = self.reparameterize(mu, logvar)       # Sample from posterior
        pi, mu_gmm, sigma_gmm = self.decoder(z, zs)  # Decode GMM parameters
        return pi, mu_gmm, sigma_gmm, mu, logvar

# GMM Negative Log-Likelihood Loss
def gmm_nll(pi, mu, sigma, expert_action):
    batch_size, num_modes, action_dim = mu.shape
    expert_action = expert_action.unsqueeze(1).repeat(1, num_modes, 1)  # Expand to match GMM shape

    # Gaussian log likelihood
    log_prob = -0.5 * (((expert_action - mu) / sigma) ** 2 + 2 * torch.log(sigma + 1e-8) + torch.log(2 * torch.pi))
    log_prob = log_prob.sum(dim=-1)  # Sum over action dimensions

    # Log-sum-exp trick for numerical stability
    weighted_log_prob = torch.log(pi + 1e-8) + log_prob  # Log mixture weights + log prob
    nll = -torch.logsumexp(weighted_log_prob, dim=-1).mean()  # Negative log likelihood
    return nll

# ELBO Loss Function
def elbo_loss(pi, mu_gmm, sigma_gmm, expert_action, mu_latent, logvar_latent, beta):
    # Reconstruction loss (GMM Negative Log-Likelihood)
    recon_loss = gmm_nll(pi, mu_gmm, sigma_gmm, expert_action)
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp())
    return recon_loss + beta * kl_loss

# Training Loop
def train_cvae(model, data_loader, optimizer, device, beta, max_iterations=100):
    model.train()
    for batch_idx, (state, action_seq, expert_action) in enumerate(data_loader):
        state = state.to(device)
        action_seq = action_seq.to(device)
        expert_action = expert_action.to(device)

        optimizer.zero_grad()
        pi, mu_gmm, sigma_gmm, mu_latent, logvar_latent = model(state, action_seq)
        loss = elbo_loss(pi, mu_gmm, sigma_gmm, expert_action, mu_latent, logvar_latent, beta)
        loss.backward()
        optimizer.step()

        if batch_idx >= max_iterations:
            break

# Hyperparameters
state_dim = 30               # State dimension
action_dim = 10              # Action dimension
latent_dim_state = 50        # Latent state dimension
latent_dim_action = 50       # Latent action dimension
posterior_dim = 64           # Posterior Gaussian dimension
rnn_hidden_dim = 1000        # RNN hidden dimension
rnn_num_layers = 2           # Number of RNN layers
gmm_num_modes = 5            # Number of GMM modes
learning_rate = 3e-4         # Learning rate
beta = 1                     # KL coefficient
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer
model = CVAEWithRNN(
    state_dim=state_dim,
    action_dim=action_dim,
    latent_dim_state=latent_dim_state,
    latent_dim_action=latent_dim_action,
    posterior_dim=posterior_dim,
    rnn_hidden_dim=rnn_hidden_dim,
    rnn_num_layers=rnn_num_layers,
    gmm_num_modes=gmm_num_modes,
    rnn_type="LSTM"
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Example data loader (needs to be defined)
# data_loader = ... (state, action_seq, expert_action)

# Train model
# train_cvae(model, data_loader, optimizer, device, beta)
