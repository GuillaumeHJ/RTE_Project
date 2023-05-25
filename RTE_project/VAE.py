import torch

import torch.nn.functional as F
import torch.nn as nn
import torch.distributions
from tqdm import tqdm
import Load_data
from matplotlib import pyplot as plt
import numpy as np


path = "data/"

train_dataloader, val_dataloader, test_dataloader, training_set, validation_set, test_set = Load_data.load(path,
                                                                                                           False)
train_cond_dataloader, val_cond_dataloader, test_cond_dataloader, training_cond, validation_cond, test_cond = Load_data.load(
    path, True)


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(48, 24)
        self.linear2 = nn.Linear(24, 12)
        self.linear3 = nn.Linear(12, latent_dims)
        self.linear4 = nn.Linear(12, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.linear3(x)
        sigma = torch.exp(self.linear4(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 24)
        self.linear2 = nn.Linear(24, 48)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def evaluate(autoencoder, x_train, x_val):
    train_loss = 0
    train_num_sample = 0
    val_loss = 0
    val_num_sample = 0
    with torch.no_grad():
        for x in x_train:
            decoded_x = autoencoder.forward(x)
            train_loss += ((x - decoded_x) ** 2).sum() + autoencoder.encoder.kl
            train_num_sample += len(x)
        for x in x_val:
            decoded_x = autoencoder.forward(x)
            val_loss += ((x - decoded_x) ** 2).sum() + autoencoder.encoder.kl
            val_num_sample += len(x)
    return train_loss / train_num_sample, val_loss / val_num_sample


def train(autoencoder, x_train, x_val, epochs=20, lr=1e-3):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    train_loss = []
    val_loss = []
    for epoch in tqdm(range(epochs)):
        for x in x_train:
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
        tl, vl = evaluate(autoencoder, x_train, x_val)
        train_loss.append(tl)
        val_loss.append(vl)
    return train_loss, val_loss


def plot_training(vae, lr, epochs, conditioned=True):
    l, v = train(vae, train_dataloader, val_dataloader, epochs=epochs, lr=lr)
    plt.plot(np.arange(len(l)), l, label='train')
    plt.plot(np.arange(len(v)), v, label='val')
    plt.title("VAE")
    plt.legend()
    plt.show()
    n = 50
    if conditioned:
        plt.plot(np.arange(48), vae(validation_cond[n].unsqueeze(0)).detach().squeeze(0).numpy())
        plt.plot(np.arange(48), validation_set[n].unsqueeze(0).detach().squeeze(0).numpy())
        plt.title("Conditional VAE")
    else:
        plt.plot(np.arange(48), vae(validation_set[n].unsqueeze(0)).detach().squeeze(0).numpy())
        plt.plot(np.arange(48), validation_set[n].unsqueeze(0).detach().squeeze(0).numpy())
        plt.title("Classic VAE")

    plt.show()


vae = VariationalAutoencoder(4)
plot_training(vae, 1e-4, 100, False)
