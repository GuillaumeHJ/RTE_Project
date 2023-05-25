import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import Load_data
import matplotlib.pyplot as plt
import numpy as np

path = "data/"

train_dataloader, val_dataloader, test_dataloader, training_set, validation_set, test_set = Load_data.load(path, False)
train_cond_dataloader, val_cond_dataloader, test_cond_dataloader, training_cond, validation_cond, test_cond = Load_data.load(path, True)


class PCA:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.proj = None
        self.mu = None

    def fit(self, X):
        self.mu = torch.mean(X, axis=0)
        _, _, v = torch.svd(X - self.mu[None, :])
        self.proj = v[:self.latent_dim, :].T

    def encode(self, X):
        return torch.matmul(X - self.mu, self.proj)

    def decode(self, encoded):
        return torch.matmul(encoded, self.proj.T)+self.mu[None,:]
class VE(nn.Module):
    def __init__(self,dim1, dim2, dim3, latent_dim):
        super().__init__()
        self.bloc = nn.Sequential(nn.Linear(48, dim1),
                                  nn.ReLU(),
                                  nn.Linear(dim1, dim2),
                                  nn.ReLU(),
                                  nn.Linear(dim2, dim3),
                                  nn.ReLU())
        self.fc_mu = nn.Linear(dim3, latent_dim)
        self.fc_var = nn.Linear(dim3, latent_dim)
        self.kl = 0 #torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        self.normal = torch.distributions.Normal(0, 1)

    def forward(self,x):
        x = self.bloc(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        n = self.normal.sample(mu.shape)
        std = torch.exp(log_var/2)
        self.kl = torch.sum(-1/2 - log_var + mu ** 2 + log_var.exp())
        return mu + n * std


class VAE(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, dim5, latent_dim):
        super().__init__()
        self.encoder = VE(dim1, dim2, dim3, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, dim4),
                                     nn.ReLU(),
                                     nn.Linear(dim4, dim5),
                                     nn.ReLU(),
                                     nn.Linear(dim5, 48))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def evaluate(self):
        train_loss = 0
        train_num_sample = 0
        val_loss = 0
        val_num_sample = 0
        with torch.no_grad():
            for x in train_dataloader:
                decoded_x = self.forward(x)
                train_loss += ((x - decoded_x) ** 2).sum() + self.encoder.kl
                train_num_sample += len(x)
            for x in val_dataloader:
                decoded_x = self.forward(x)
                val_loss += ((x - decoded_x) ** 2).sum() + self.encoder.kl
                val_num_sample += len(x)
        return train_loss / train_num_sample, val_loss / val_num_sample

    def train(self, epoch, lr):
        loss_function = nn.MSELoss(reduce=True, reduction='sum')
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        TL = []
        VL = []
        for i in tqdm(range(epoch)):
            for t, x in enumerate(train_dataloader):
                opt.zero_grad
                decoded_x = self.forward(x)
                loss = ((x - decoded_x) ** 2).sum() + self.encoder.kl
                loss.backward()
                opt.step()
            tl, vl = self.evaluate()
            TL.append(tl)
            VL.append(vl)
        return TL, VL

"""
pca = PCA(4)
pca.fit(training_set)
latent_space = pca.encode(training_cond[:,:48])

plt.scatter(latent_space[:,0], latent_space[:,1], c=training_cond[:,49], cmap='viridis')
plt.title("pca 0-1")
plt.show()

x_hat = pca.decode(latent_space)
plt.plot(np.arange(48), x_hat[0,:].detach().numpy())
plt.show()
"""
latent_space_dim = 4
dim1 = 30; dim2 = 20; dim3 = 10 ; dim4 = 10; dim5 = 20
lr =5*1e-5
epochs = 200

autoencoder = VAE(dim1, dim2, dim3,dim4, dim5, latent_space_dim)

l, v = autoencoder.train(epochs, lr)
plt.plot(np.arange(len(l)), l)
plt.plot(np.arange(len(v)), v)
plt.show()

latent_space = autoencoder.encoder(training_set).detach().numpy()
plt.scatter(latent_space[:, 0], latent_space[:, 1], c=training_cond[:, 49], cmap='viridis')
plt.title('0-1')
plt.show()
plt.scatter(latent_space[:, 1], latent_space[:, 2], c=training_cond[:, 49], cmap='viridis')
plt.title('0-2')
plt.show()

plt.scatter(latent_space[:, 2], latent_space[:, 0], c=training_cond[:, 49], cmap='viridis')
plt.title('0-3')
plt.show()

x_hat = autoencoder(training_set[100].unsqueeze(0)).squeeze()
plt.plot(np.arange(48), x_hat[:].detach().numpy())
plt.plot(np.arange(48), training_set[100])
plt.show()