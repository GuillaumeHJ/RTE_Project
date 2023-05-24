import matplotlib.pyplot as plt
import numpy as np

import ClementVAE

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def regression_temp(encoder, x_train):
    latent_space, _ = encoder(x_train)
    latent_space = np.array(latent_space)
    y = np.array(x_train[:, 48])
    X_train, X_test, y_train, y_test = train_test_split(latent_space, y, random_state=1)
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    latent_score = regr.score(X_test, y_test)

    X_train, X_test, y_train, y_test = train_test_split(x_train[:, :48], y, random_state=1)
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    original_score = regr.score(X_test, y_test)

    return original_score, latent_score

print(regression_temp(ClementVAE.encoder, ClementVAE.x_train))

def classifier_day(encoder, x_train):
    latent_space, _ = encoder(x_train)
    latent_space = np.array(latent_space)
    y = np.array(x_train[:, 49])
    X_train, X_test, y_train, y_test = train_test_split(latent_space, y, random_state=1)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    latent_score = clf.score(X_test, y_test)

    X_train, X_test, y_train, y_test = train_test_split(x_train[:, :48], y, random_state=1)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    original_score = clf.score(X_test, y_test)

    return original_score, latent_score


print(classifier_day(ClementVAE.encoder, ClementVAE.x_train))

def scatter_plot_latent_space(encoder, x_train, dx=0, dy=1):
    latent_space, _ = encoder(x_train)
    plt.scatter(latent_space[:, dx], latent_space[:, dy], c=x_train[:, 49], cmap='viridis', marker='.')
    plt.title(f'Scatterplot of dimension {dy} with respect to dimension {dx}')
    plt.colorbar(label='Day of the week')
    plt.show()

    plt.scatter(latent_space[:, dx], latent_space[:, dy], c=x_train[:, 48], cmap='gist_heat', marker='.')
    plt.title(f'Scatterplot of dimension {dy} with respect to dimension {dx}')
    plt.colorbar(label='Temperature')
    plt.show()


scatter_plot_latent_space(ClementVAE.encoder, ClementVAE.x_train, 0, 1)

def plot_pca(x_train):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(x_train[:, :48])
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=x_train[:, 49], cmap='viridis', marker='.')
    plt.title('PCA')
    plt.colorbar(label='Day of the week')
    plt.show()

plot_pca(ClementVAE.x_train)
