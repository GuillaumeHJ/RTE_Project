import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import Load_data
from matplotlib import pyplot as plt
import numpy as np

path = "data/"

train_dataloader, val_dataloader, test_dataloader, training_set, validation_set, test_set = Load_data.load(path,
                                                                                                              False)
train_cond_dataloader, val_cond_dataloader, test_cond_dataloader, training_cond, validation_cond, test_cond = Load_data.load(
    path, True)



# **************************************Building a VAE***************************************

class VariationalEncoder(nn.Module):
    def __init__(self, latent_space_dim, hidden_layer_dim):
        super().__init__()

        self.fc = [nn.Linear(48, hidden_layer_dim[0])]
        for i in range(len(hidden_layer_dim) - 1):
            self.fc.append(nn.Linear(hidden_layer_dim[i], hidden_layer_dim[i + 1]))
        # for linear in self.fc:
        #   nn.init.kaiming_normal_(linear.weight)
        self.var = nn.Linear(hidden_layer_dim[-1], latent_space_dim)
        self.mean = nn.Linear(hidden_layer_dim[-1], latent_space_dim)
        self.normal = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
            x = F.relu(x)
        log_var = self.var(x)
        mu = self.mean(x)
        encoded = mu + torch.exp(log_var / 2) * self.normal.sample(mu.shape)
        # self.kl = torch.sum(s**2 + m**2 - torch.log(1e-8 + s) - 1/2) ##KL divergence between the standard gaussian distribution and the generated distribution
        self.kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return encoded


class Decoder(nn.Module):
    def __init__(self, latent_space_dim, hidden_layer_dim, cond=False):
        super().__init__()
        cond = int(cond)
        self.fc = [nn.Linear(latent_space_dim + cond, hidden_layer_dim[-1])]
        for i in range(len(hidden_layer_dim) - 1, 0, -1):
            self.fc.append(nn.Linear(hidden_layer_dim[i], hidden_layer_dim[i - 1]))

        self.fc.append(nn.Linear(hidden_layer_dim[0], 48))
        for linear in self.fc:
            nn.init.kaiming_normal_(linear.weight)

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = F.relu(x)
        x = self.fc[-1](x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_space_dim, hidden_layer_dim):
        super().__init__()
        self.encoder = VariationalEncoder(latent_space_dim, hidden_layer_dim)
        self.decoder = Decoder(latent_space_dim, hidden_layer_dim, cond=False)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def latent(self, x):
        return self.encoder(x[:, :48])

    def evaluate(self):
        train_loss = 0
        train_num_sample = 0
        val_loss = 0
        val_num_sample = 0
        with torch.no_grad():
            for x in train_dataloader:
                decoded_x = self.forward(x)
                train_loss += ((x - decoded_x) ** 2).sum() / (x.shape[0]) + self.encoder.kl
                train_num_sample += len(x)
            for x in val_dataloader:
                decoded_x = self.forward(x)
                val_loss += ((x - decoded_x) ** 2).sum() / (x.shape[0]) + self.encoder.kl
                val_num_sample += len(x)
        return train_loss / train_num_sample, val_loss / val_num_sample

    def train(self, epoch, lr):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        TL = []
        VL = []
        for i in tqdm(range(epoch)):
            for t, x in enumerate(train_dataloader):
                opt.zero_grad
                decoded_x = self.forward(x)
                loss = ((x - decoded_x) ** 2).sum() / (x.shape[0]) + self.encoder.kl
                loss.backward()
                opt.step()
            tl, vl = self.evaluate()
            TL.append(tl)
            VL.append(vl)
        return TL, VL


# *****************************Conditionned VAE modules***************************************


class CondDecoder(nn.Module):
    def __init__(self, latent_space_dim, hidden_layer_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_space_dim + 1, hidden_layer_dim[2])
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_layer_dim[2], hidden_layer_dim[1])
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(hidden_layer_dim[1], hidden_layer_dim[0])
        nn.init.kaiming_normal_(self.fc3.weight)
        self.decoder_output_layer = nn.Linear(hidden_layer_dim[0], 48)
        nn.init.kaiming_normal_(self.decoder_output_layer.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.decoder_output_layer(x)
        return x


class CondVAE(nn.Module):
    def __init__(self, latent_space_dim, hidden_layer_dim):
        super().__init__()
        self.encoder = VariationalEncoder(latent_space_dim, hidden_layer_dim)
        self.decoder = CondDecoder(latent_space_dim, hidden_layer_dim)

    def forward(self, x):
        encoded = self.encoder(x[:, :48])
        cond_x = torch.cat((encoded, x[:, 48, None]), dim=1)
        return self.decoder(cond_x)

    def latent(self, x):
        return self.encoder(x[:, :48])

    def evaluate(self):
        train_loss = 0
        train_num_sample = 0
        val_loss = 0
        val_num_sample = 0
        with torch.no_grad():
            for x in train_cond_dataloader:
                decoded_x = self.forward(x)
                train_loss += ((x[:, :48] - decoded_x) ** 2).sum() / (x.shape[0] - 1) + self.encoder.kl
                train_num_sample += len(x)
            for x in val_cond_dataloader:
                decoded_x = self.forward(x)
                val_loss += ((x[:, :48] - decoded_x) ** 2).sum() / (x.shape[0] - 1) + self.encoder.kl
                val_num_sample += len(x)
        return train_loss / train_num_sample, val_loss / val_num_sample

    def train(self, epoch, lr):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        TL = []
        VL = []
        for i in tqdm(range(epoch)):
            for t, x in enumerate(train_cond_dataloader):
                opt.zero_grad
                decoded_x = self.forward(x)
                loss = ((x[:, :48] - decoded_x) ** 2).sum() / (x.shape[0] - 1) + self.encoder.kl
                loss.backward()
                opt.step()
            tl, vl = self.evaluate()
            TL.append(tl)
            VL.append(vl)
        return TL, VL


# hyper parameters
latent_space_dim = 5
hidden_layer_dim = [250, 120, 60]
lr = 1e-4
epochs = 20



def plot_training(vae, lr, epochs, conditioned=True):

    l, v = vae.train(epochs, lr)
    plt.plot(np.arange(len(l)), l, label='train')
    plt.plot(np.arange(len(v)), v, label='val')
    plt.title("VAE")
    plt.legend()
    plt.show()
    n= 50
    if conditioned:
        plt.plot(np.arange(48), vae(validation_cond[n].unsqueeze(0)).detach().squeeze(0).numpy())
        plt.plot(np.arange(48), validation_set[n].unsqueeze(0).detach().squeeze(0).numpy())
        plt.title("Conditional VAE")
    else:
        plt.plot(np.arange(48), vae(validation_set[n].unsqueeze(0)).detach().squeeze(0).numpy())
        plt.plot(np.arange(48), validation_set[n].unsqueeze(0).detach().squeeze(0).numpy())
        plt.title("Classic VAE")

    plt.show()

#vae = CondVAE(latent_space_dim, hidden_layer_dim)
#plot_training(vae, lr, epochs, True)