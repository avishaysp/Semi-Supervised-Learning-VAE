import torch
import torch.nn as nn

HIDDEN_DIM = 600


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(HIDDEN_DIM, latent_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, input_dim)
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(vae, dataloader, optimizer, device, epochs=10):
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device).view(x.size(0), -1)
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader.dataset):.4f}")

def extract_latent_features(vae, dataloader, device):
    vae.eval()
    latent_features = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device).view(x.size(0), -1) 
            mu, _ = vae.encode(x)
            latent_features.append(mu.cpu())
            labels.append(y)
    return torch.cat(latent_features).numpy(), torch.cat(labels).numpy()