import matplotlib.pyplot as plt

import Load_data
import VAE_model

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

path = "data/"

train_dataloader, val_dataloader, test_dataloader, training_set, validation_set, test_set = Load_data.load(path,
                                                                                                           False)
train_cond_dataloader, val_cond_dataloader, test_cond_dataloader, training_cond, validation_cond, test_cond = Load_data.load(
    path, True)

# hyper parameters
latent_space_dim = 4
hidden_layer_dim = [30, 20, 10]
lr = 5*1e-5
epochs = 100

vae_cond = VAE_model.CondVAE(latent_space_dim, hidden_layer_dim)
vae = VAE_model.VAE(latent_space_dim, hidden_layer_dim)

l, v = vae_cond.train(epochs, lr)
l, v = vae.train(epochs, lr)


def regression_temp(trained_model, training_cond):
    X = trained_model.latent(training_cond[:, :48]).detach().numpy()
    y = training_cond[:, 48].detach().numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    latent_score = regr.score(X_test, y_test)

    X_train, X_test, y_train, y_test = train_test_split(training_cond[:, :48], y, random_state=1)
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    original_score = regr.score(X_test, y_test)

    return original_score, latent_score


print(regression_temp(vae_cond, training_cond))
print(regression_temp(vae, training_cond))

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


print(classifier_day(vae_cond, training_cond))
print(classifier_day(vae, training_cond))

def scatter_plot_latent_space(trained_model, training_cond, dx=0, dy=1):
    latent_space = trained_model.latent(training_cond[:, :48]).detach().numpy()
    plt.scatter(latent_space[:, dx], latent_space[:, dy], c=training_cond[:, 49], cmap='viridis')
    plt.title(f'Scatterplot of dimension {dy} with respect to dimension {dx} of {trained_model.__class__.__name__}')
    plt.colorbar(label='Day of the week')
    plt.show()

    plt.scatter(latent_space[:, dx], latent_space[:, dy], c=training_cond[:, 48], cmap='gist_heat')
    plt.title(f'Scatterplot of dimension {dy} with respect to dimension {dx} of {trained_model.__class__.__name__}')
    plt.colorbar(label='Temperature')
    plt.show()

scatter_plot_latent_space(vae, training_cond, 0, 1)
scatter_plot_latent_space(vae_cond, training_cond, 0, 1)
scatter_plot_latent_space(vae, training_cond, 1, 2)
scatter_plot_latent_space(vae_cond, training_cond, 1, 2)
scatter_plot_latent_space(vae, training_cond, 2, 3)
scatter_plot_latent_space(vae_cond, training_cond, 2, 3)

def plot_pca(trained_model, training_cond):
    latent_space = trained_model.latent(training_cond[:, :48]).detach().numpy()
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(latent_space)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=training_cond[:, 49], cmap='viridis')
    plt.title('PCA')
    plt.colorbar(label='Day of the week')
    plt.show()

plot_pca(vae_cond, training_cond)
