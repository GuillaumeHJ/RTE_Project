import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import New_load
from torch.utils.data import DataLoader

path = "data/"

X_train, X_val, sc, weekdays, months, df_prev_val = New_load.load(path)

train_dataloader = DataLoader(X_train, batch_size=64, shuffle=True)
val_dataloader = DataLoader(X_val, batch_size=64, shuffle=True)


# **************************************Building Variationnal Encoder and Decoder***************************************

class VariationalEncoder(nn.Module):
    def __init__(self, latent_space_dim, hidden_layer_dim):
        super().__init__()

        self.fc = nn.ModuleList([nn.Linear(48, hidden_layer_dim[0])])
        for i in range(len(hidden_layer_dim) - 1):
            self.fc.append(nn.Linear(hidden_layer_dim[i], hidden_layer_dim[i + 1]))
            self.fc.append(nn.ReLU())
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        self.var = nn.Linear(hidden_layer_dim[-1], latent_space_dim)
        self.mean = nn.Linear(hidden_layer_dim[-1], latent_space_dim)
        self.normal = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        log_var = self.var(x)
        mu = self.mean(x)
        encoded = mu + torch.exp(log_var / 2) * self.normal.sample(mu.shape)
        self.kl = torch.sum(-1 / 2 - log_var + mu ** 2 + log_var.exp())
        return encoded


class Decoder(nn.Module):
    def __init__(self, latent_space_dim, hidden_layer_dim, cond=False):
        super().__init__()
        cond = int(cond)
        self.fc = nn.ModuleList([nn.Linear(latent_space_dim + 8 * cond, hidden_layer_dim[0])])
        self.fc.append(nn.ReLU())
        for i in range(len(hidden_layer_dim) - 1):
            self.fc.append(nn.Linear(hidden_layer_dim[i], hidden_layer_dim[i + 1]))
            self.fc.append(nn.ReLU())
        self.fc.append(nn.Linear(hidden_layer_dim[-1], 48))
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        return x


# ***************************** VAE ***************************************


class VAE(nn.Module):
    def __init__(self, latent_space_dim, hidden_layer_encoder, hidden_layer_decoder, conditioned=False):
        super().__init__()
        self.encoder = VariationalEncoder(latent_space_dim, hidden_layer_encoder)
        self.decoder = Decoder(latent_space_dim, hidden_layer_decoder, conditioned)
        self.conditioned = conditioned

    def forward(self, x):
        encoded = self.encoder(x[:, :48])
        if self.conditioned:
            week_day_one_hot = F.one_hot(x[:, 49].long(), num_classes=7)
            encoded = torch.cat((encoded, x[:, 48, None], week_day_one_hot), dim=1)
        return self.decoder(encoded)

    def latent(self, x):
        return self.encoder(x[:, :48])

    def evaluate(self, train_dataloader, val_dataloader):
        train_loss = 0
        train_num_sample = 0
        val_loss = 0
        val_num_sample = 0
        with torch.no_grad():
            for x in train_dataloader:
                if self.conditioned:
                    decoded_x = self.forward(x)
                else:
                    decoded_x = self.forward(x[:, :48])
                train_loss += ((x[:, :48] - decoded_x) ** 2).sum() + self.encoder.kl
                train_num_sample += len(x)
            for x in val_dataloader:
                if self.conditioned:
                    decoded_x = self.forward(x)
                else:
                    decoded_x = self.forward(x[:, :48])
                val_loss += ((x[:, :48] - decoded_x) ** 2).sum() + self.encoder.kl
                val_num_sample += len(x)
        return train_loss / train_num_sample, val_loss / val_num_sample

    def train(self, epoch, lr, train_dataloader, val_dataloader):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        TL = []
        VL = []
        for i in tqdm(range(epoch)):
            for t, x in enumerate(train_dataloader):
                opt.zero_grad
                if self.conditioned:
                    decoded_x = self.forward(x)
                else:
                    decoded_x = self.forward(x[:, :48])
                loss = ((x[:, :48] - decoded_x) ** 2).sum() + self.encoder.kl
                loss.backward()
                opt.step()
            tl, vl = self.evaluate(train_dataloader, val_dataloader)
            TL.append(tl)
            VL.append(vl)
        return TL, VL


# hyper parameters
latent_space_dim = 3
hidden_layer_encoder = [48, 48, 24, 12]
hidden_layer_decoder = [12, 24]
lr = 5 * 1e-5
epochs = 500


def plot_training(vae, lr, epochs):
    l, v = vae.train(epochs, lr, train_dataloader, val_dataloader)
    print(v)
    conditioned = vae.conditioned
    plt.plot(np.arange(len(l)), l, label='train')
    plt.plot(np.arange(len(v)), v, label='val')
    plt.title("Training Losses")
    plt.legend()
    plt.show()
    n = 200
    if conditioned:
        plt.plot(np.arange(48), New_load.descale(vae(torch.Tensor(X_val[n]).unsqueeze(0)).detach(), sc).squeeze(0),
                 label='Genrated Scenario')
        plt.plot(np.arange(48), New_load.descale(X_val[:, :48], sc)[n], label='Real Profile')
        plt.legend()
        plt.title("Conditional VAE")
    else:

        plt.plot(np.arange(48), New_load.descale(vae(torch.Tensor(X_val[n][:48]).unsqueeze(0)).detach(), sc).squeeze(0),
                 label='Genrated Scenario')
        plt.plot(np.arange(48), New_load.descale(X_val[:, :48], sc)[n], label='Real Profile')
        plt.title("Classic VAE")

    plt.show()


#############################" Code to plot training and scenario generated from random sample #################################
# n = 200
# sample = np.random.normal(size=(1, latent_space_dim))
# sample = torch.Tensor(sample)
# week_day_one_hot = F.one_hot(torch.Tensor(X_val[n:n+1, 49]).long(), num_classes=7)
# cond_sample = torch.cat((sample, torch.tensor(X_val[n:n+1, 48, None]), week_day_one_hot), dim=1)
#
# vae = VAE(latent_space_dim, hidden_layer_encoder, hidden_layer_decoder, conditioned=True)
# plot_training(vae, lr, epochs)
# plt.plot(np.arange(48), np.squeeze(New_load.descale(
#     vae.decoder(cond_sample).detach()[:, :48], sc)),
#          label='Generated Scenario')
# plt.plot(np.arange(48), New_load.descale(X_val[:, :48], sc)[n], label='Real Profile')
# plt.legend()
# plt.show()
