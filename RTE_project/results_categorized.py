import matplotlib.pyplot as plt
import numpy as np
import New_load
import VAE_model
from torch.utils.data import DataLoader
import scoring_pytorch as scoring
from tqdm import tqdm


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)

clement = False
M = 100
latent_space_dim = 3

path = "data/"

X_train, X_val, sc, weekdays, months, df_prev_val = New_load.load(path, days=True, month=True)

train_dataloader = DataLoader(X_train, batch_size=64, shuffle=True)
val_dataloader = DataLoader(X_val, batch_size=64, shuffle=True)

weekdays_loader = []
months_loader = []

# for day in weekdays:
#     print(day.shape)
#
# for month in months:
#     print(month.shape)

# hyper parameters
hidden_layer_encoder = [48, 48, 24, 12]
hidden_layer_decoder = [12, 24]
lr = 5 * 1e-5
epochs = 200

vae_cond = VAE_model.VAE(latent_space_dim, hidden_layer_encoder, hidden_layer_decoder, conditioned=True)
l, v = vae_cond.train(epochs, lr, train_dataloader, val_dataloader)

########################### TEMPERATURE #####################################

# Diviser le dataset en 10 sous-datasets correspondant à des plages de valeurs distinctes
num_slices = 10
temp_slices = [[] for _ in range(num_slices)]

# Obtenir les indices triés en fonction de la 48ème coordonnée de l'axe 1
sorted_indices = np.argsort(X_val[:, 48])

# Trier le tableau en fonction des indices triés
sorted_X_val = X_val[sorted_indices]
temp_values = New_load.descale(sorted_X_val, sc)[:, 48]

min_val = np.min(temp_values)
max_val = np.max(temp_values)
gap = (max_val - min_val) / num_slices
value_ranges = [[min_val + i * gap, min_val + (i + 1) * gap] for i in
                range(num_slices)]

for i, val_range in enumerate(value_ranges):
    for n in range(X_val.shape[0]):
        if val_range[0] <= temp_values[n] < val_range[1]:
            temp_slices[i].append(X_val[n])
    temp_slices[i] = np.array(temp_slices[i])




temp_score = np.zeros(shape=(4, num_slices))

for i, slice in tqdm(enumerate(temp_slices)):
    _, median, _, scenarios = scoring.generate_scenarios(vae_cond.decoder, slice, latent_space_dim=latent_space_dim,
                                                         sc=sc, M=M)
    temp_score[0, i] = scoring.energy_score(New_load.descale(slice, sc), scenarios)
    temp_score[1, i] = scoring.variogram_score(New_load.descale(slice, sc), scenarios)
    temp_score[2, i] = scoring.quantile_score(New_load.descale(slice, sc), scenarios)
    temp_score[3, i] = scoring.MSE(New_load.descale(slice, sc)[:, :48], median[:, :48])

# Créer une figure et des sous-graphiques pour chaque élément de l'axe 0 de weekday_score
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 10))

# Parcourir les éléments de l'axe 0 de weekday_score
for i in range(4):
    # Obtenir le i-ème élément de l'axe 0 et les valeurs correspondantes
    data = temp_score[i]

    # Créer un histogramme pour l'élément i avec les jours de la semaine en abscisse
    value_ranges = [[trunc(min_val + i * gap, decs=0), trunc(min_val + (i + 1) * gap, decs=0)] for i in
                    range(num_slices)]
    value_ranges_names = [str(val_range) for val_range in value_ranges]
    axs[i].bar(value_ranges_names, data)
    if i == 0:
        axs[i].set_title('Histogram of Energy Score')
    elif i == 1:
        axs[i].set_title('Histogram of Variogram Score')
    elif i == 2:
        axs[i].set_title('Histogram of Quantile Score')
    elif i == 3:
        axs[i].set_title('Histogram of MSE Score')
    axs[i].set_xlabel('Temperature')
    axs[i].set_ylabel('Value')

    # Obtenir la valeur maximale de l'élément i pour ajuster les limites de l'axe des ordonnées
    max_value = np.max(data)
    axs[i].set_ylim([0, max_value + max_value * 0.03])  # Ajuster les limites de l'axe des ordonnées avec un zoom de 10%

# Ajuster les espaces entre les sous-graphiques
plt.tight_layout()

# Afficher les histogrammes
plt.show()

############################################## WEEKDAYS #########################################

weekdays_score = np.zeros(shape=(4, 7))

for i, day in tqdm(enumerate(weekdays)):
    _, median, _, scenarios = scoring.generate_scenarios(vae_cond.decoder, day, latent_space_dim=latent_space_dim,
                                                         sc=sc, M=M)
    weekdays_score[0, i] = scoring.energy_score(New_load.descale(day, sc), scenarios)
    weekdays_score[1, i] = scoring.variogram_score(New_load.descale(day, sc), scenarios)
    weekdays_score[2, i] = scoring.quantile_score(New_load.descale(day, sc), scenarios)
    weekdays_score[3, i] = scoring.MSE(New_load.descale(day, sc)[:, :48], median[:, :48])

# Définir les noms des jours de la semaine
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Créer une figure et des sous-graphiques pour chaque élément de l'axe 0 de weekday_score
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 10))

# Parcourir les éléments de l'axe 0 de weekday_score
for i in range(4):
    # Obtenir le i-ème élément de l'axe 0 et les valeurs correspondantes
    data = weekdays_score[i]

    # Créer un histogramme pour l'élément i avec les jours de la semaine en abscisse
    axs[i].bar(weekday_names, data)
    if i == 0:
        axs[i].set_title('Histogram of Energy Score')
    elif i == 1:
        axs[i].set_title('Histogram of Variogram Score')
    elif i == 2:
        axs[i].set_title('Histogram of Quantile Score')
    elif i == 3:
        axs[i].set_title('Histogram of MSE Score')
    axs[i].set_xlabel('Weekdays')
    axs[i].set_ylabel('Value')

    # Obtenir la valeur maximale de l'élément i pour ajuster les limites de l'axe des ordonnées
    max_value = np.max(data)
    axs[i].set_ylim([0, max_value + max_value * 0.03])  # Ajuster les limites de l'axe des ordonnées avec un zoom de 10%

# Ajuster les espaces entre les sous-graphiques
plt.tight_layout()

# Afficher les histogrammes
plt.show()

############################################ MONTHS ############################################

months_score = np.zeros(shape=(4, 12))

for i, month in tqdm(enumerate(months)):
    _, median, _, scenarios = scoring.generate_scenarios(vae_cond.decoder, month, latent_space_dim=latent_space_dim,
                                                         sc=sc, M=M)
    months_score[0, i] = scoring.energy_score(New_load.descale(month, sc), scenarios)
    months_score[1, i] = scoring.variogram_score(New_load.descale(month, sc), scenarios)
    months_score[2, i] = scoring.quantile_score(New_load.descale(month, sc), scenarios)
    months_score[3, i] = scoring.MSE(New_load.descale(month, sc)[:, :48], median[:, :48])

# Définir les noms des mois de l'année
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Créer une figure et des sous-graphiques pour chaque élément de l'axe 0 de weekday_score
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 10))

# Parcourir les éléments de l'axe 0 de weekday_score
for i in range(4):
    # Obtenir le i-ème élément de l'axe 0 et les valeurs correspondantes
    data = months_score[i]

    # Créer un histogramme pour l'élément i avec les noms des mois en abscisse
    axs[i].bar(month_names, data)
    if i == 0:
        axs[i].set_title('Histogram of Energy Score')
    elif i == 1:
        axs[i].set_title('Histogram of Variogram Score')
    elif i == 2:
        axs[i].set_title('Histogram of Quantile Score')
    elif i == 3:
        axs[i].set_title('Histogram of MSE Score')
    axs[i].set_xlabel('Months')
    axs[i].set_ylabel('Value')

    # Obtenir la valeur maximale de l'élément i pour ajuster les limites de l'axe des ordonnées
    max_value = np.max(data)
    axs[i].set_ylim([0, max_value + max_value * 0.03])  # Ajuster les limites de l'axe des ordonnées avec un zoom de 10%

# Ajuster les espaces entre les sous-graphiques
plt.tight_layout()

# Afficher les histogrammes
plt.show()

_, median, _, scenarios = scoring.generate_scenarios(vae_cond.decoder, X_val, latent_space_dim=latent_space_dim,
                                                     sc=sc, M=M)

energy_temp = scoring.energy_score(X_val, scenarios)
variogram_temp = scoring.variogram_score(X_val, scenarios)
quantile_temp = scoring.quantile_score(X_val, scenarios)
MSE_temp = scoring.MSE(X_val[:, :48], median[:, :48])
