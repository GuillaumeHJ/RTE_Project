import matplotlib.pyplot as plt
import numpy as np
import New_load
import VAE_model
from torch.utils.data import DataLoader
import scoring_pytorch as scoring

clement = False
M = 100
latent_space_dim = 3

path = "data/"

X_train, X_val, sc, weekdays, months = New_load.load(path, days=True, month=True)

train_dataloader = DataLoader(X_train, batch_size=64, shuffle=True)
val_dataloader = DataLoader(X_val, batch_size=64, shuffle=True)

weekdays_loader = []
months_loader = []

for day in weekdays:
    weekdays_loader.append(DataLoader(day, batch_size=64, shuffle=True))

for month in months:
    months_loader.append(DataLoader(month, batch_size=64, shuffle=True))

# hyper parameters
hidden_layer_encoder = [48, 48, 24, 12]
hidden_layer_decoder = [12, 24]
lr = 5 * 1e-5
epochs = 200

vae_cond = VAE_model.VAE(latent_space_dim, hidden_layer_encoder, hidden_layer_decoder, conditioned=True)
l, v = vae_cond.train(epochs, lr, train_dataloader, val_dataloader)

q1, median, q3, scenarios = scoring.generate_scenarios(vae_cond.decoder, X_val, latent_space_dim=latent_space_dim,
                                                       sc=sc, M=M)

