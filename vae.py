import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

HIDDEN_DIM = 600


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.fc_mu = nn.Linear(HIDDEN_DIM, latent_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

