import matplotlib.pyplot as plt
import numpy as np
import New_load
import VAE_model
from torch.utils.data import DataLoader
import scoring_pytorch as scoring
from tqdm import tqdm

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

weekdays_score = np.zeros(shape=(3, 7))
months_score = np.zeros(shape=(3, 12))

for i, day in tqdm(enumerate(weekdays)):
    _, _, _, scenarios = scoring.generate_scenarios(vae_cond.decoder, day, latent_space_dim=latent_space_dim,
                                                    sc=sc, M=M)
    weekdays_score[0, i] = scoring.energy_score(day, scenarios)
    weekdays_score[1, i] = scoring.variogram_score(day, scenarios)
    weekdays_score[2, i] = scoring.quantile_score(day, scenarios)

# Définir les noms des jours de la semaine
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Créer une figure et des sous-graphiques pour chaque élément de l'axe 0 de weekday_score
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))

# Parcourir les éléments de l'axe 0 de weekday_score
for i in range(3):
    # Obtenir le i-ème élément de l'axe 0 et les valeurs correspondantes
    data = weekdays_score[i]

    # Créer un histogramme pour l'élément i avec les jours de la semaine en abscisse
    axs[i].bar(weekday_names, data)
    axs[i].set_title(f'Histogram {i + 1}')
    axs[i].set_xlabel('Weekdays')
    axs[i].set_ylabel('Value')

# Ajuster les espaces entre les sous-graphiques
plt.tight_layout()

# Afficher les histogrammes
plt.show()

for i, month in tqdm(enumerate(months)):
    _, _, _, scenarios = scoring.generate_scenarios(vae_cond.decoder, month, latent_space_dim=latent_space_dim,
                                                    sc=sc, M=M)
    months_score[0, i] = scoring.energy_score(month, scenarios)
    months_score[1, i] = scoring.variogram_score(month, scenarios)
    months_score[2, i] = scoring.quantile_score(month, scenarios)

# Définir les noms des mois de l'année
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Créer une figure et des sous-graphiques pour chaque élément de l'axe 0 de weekday_score
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))

# Parcourir les éléments de l'axe 0 de weekday_score
for i in range(3):
    # Obtenir le i-ème élément de l'axe 0 et les valeurs correspondantes
    data = months_score[i]

    # Créer un histogramme pour l'élément i avec les noms des mois en abscisse
    axs[i].bar(month_names, data)
    axs[i].set_title(f'Histogram {i + 1}')
    axs[i].set_xlabel('Months')
    axs[i].set_ylabel('Value')

# Ajuster les espaces entre les sous-graphiques
plt.tight_layout()

# Afficher les histogrammes
plt.show()
