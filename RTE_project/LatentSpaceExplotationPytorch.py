import matplotlib.pyplot as plt
import torch

import New_load
import VAE_model
from torch.utils.data import DataLoader
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

path = "data/"

X_train, X_val, sc, weekdays, months = New_load.load(path)

train_dataloader = DataLoader(X_train, batch_size=64, shuffle=True)
val_dataloader = DataLoader(X_val, batch_size=64, shuffle=True)

# hyper parameters
latent_space_dim = 3
hidden_layer_encoder = [48, 48, 24, 12]
hidden_layer_decoder = [12, 24]
lr = 1e-4
epochs = 200

vae_cond = VAE_model.VAE(latent_space_dim, hidden_layer_encoder, hidden_layer_decoder, conditioned=True)
vae = VAE_model.VAE(latent_space_dim, hidden_layer_encoder, hidden_layer_decoder)

l, v = vae_cond.train(epochs, lr, train_dataloader, val_dataloader)
l, v = vae.train(epochs, lr, train_dataloader, val_dataloader)


def regression_temp(trained_model, training_cond):
    X = trained_model.latent(training_cond[:, :48]).detach().numpy()
    y = New_load.descale(training_cond[:, 48:49].detach(), sc).squeeze(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    regr = MLPRegressor(random_state=1, max_iter=300).fit(X_train, y_train)
    latent_score = regr.score(X_test, y_test)

    X_train, X_test, y_train, y_test = train_test_split(training_cond[:, :48], y, random_state=1)
    regr = MLPRegressor(random_state=1, max_iter=300).fit(X_train, y_train)
    original_score = regr.score(X_test, y_test)

    return original_score, latent_score


print(regression_temp(vae_cond, torch.Tensor(X_val)))
print(regression_temp(vae, torch.Tensor(X_val)))

def classifier_day(trained_model, training_cond):
    X = trained_model.latent(training_cond[:, :48]).detach().numpy()
    y = training_cond[:, 49].detach().numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    latent_score = clf.score(X_test, y_test)

    X_train, X_test, y_train, y_test = train_test_split(training_cond[:, :48], y, random_state=1)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    original_score = clf.score(X_test, y_test)

    return original_score, latent_score


print(classifier_day(vae_cond, torch.Tensor(X_train)))
print(classifier_day(vae, torch.Tensor(X_train)))

def scatter_plot_latent_space(trained_model, training_cond, dx=0, dy=1):
    latent_space = trained_model.latent(training_cond[:, :48]).detach().numpy()
    plt.scatter(latent_space[:, dx], latent_space[:, dy], c=training_cond[:, 49], cmap='viridis')
    plt.title(f'Scatterplot of dimension {dy} with respect to dimension {dx} with conditioning:{trained_model.conditioned}')
    plt.colorbar(label='Day of the week')
    plt.show()

    plt.scatter(latent_space[:, dx], latent_space[:, dy], c=training_cond[:, 48], cmap='gist_heat')
    plt.title(f'Scatterplot of dimension {dy} with respect to dimension {dx} with conditioning {trained_model.conditioned}')
    plt.colorbar(label='Temperature')
    plt.show()

scatter_plot_latent_space(vae, torch.Tensor(X_train), 0, 1)
scatter_plot_latent_space(vae_cond, torch.Tensor(X_train), 0, 1)
scatter_plot_latent_space(vae, torch.Tensor(X_train), 1, 2)
scatter_plot_latent_space(vae_cond, torch.Tensor(X_train), 1, 2)
scatter_plot_latent_space(vae, torch.Tensor(X_train), 2, 3)
scatter_plot_latent_space(vae_cond, torch.Tensor(X_train), 2, 3)

def plot_pca(trained_model, training_cond):
    latent_space = trained_model.latent(training_cond[:, :48]).detach().numpy()
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(latent_space)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=training_cond[:, 49], cmap='viridis')
    plt.title('PCA')
    plt.colorbar(label='Day of the week')
    plt.show()

plot_pca(vae_cond, torch.Tensor(X_train))