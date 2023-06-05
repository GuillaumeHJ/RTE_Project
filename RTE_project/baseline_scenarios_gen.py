import New_load
import numpy as np
from sklearn.decomposition import PCA


def avg_scenario(X_train, X_val, sc, M):
    n, p = X_val[:, :48].shape
    rescaled_scenarios = New_load.descale(X_train[:, :48], sc)
    avg_scenario = np.mean(rescaled_scenarios, axis=0)
    return np.broadcast_to(avg_scenario, (n, M, p))


def pca_scenario(X_train, sc, latent_space_dim, M):
    pca = PCA(n_components=latent_space_dim)
    pca.fit(X_train[:, :48])
    scenarios_pca = []
    for _ in range(M):
        sample = np.random.normal(size=(365, latent_space_dim))
        predict_pca = pca.inverse_transform(sample)
        predict_pca = New_load.descale(predict_pca, sc)
        scenarios_pca.append(predict_pca)

    return np.swapaxes(np.array(scenarios_pca), 0, 1)


def random_scenario(X_train, X_val, sc, M):
    n_train, p_train = X_train[:, :48].shape
    n_val, p_val = X_val[:, :48].shape
    scenarios_random = []
    for _ in range(M):
        index = np.random.choice(n_train, n_val, replace=False)
        sample = X_train[index, :48]
        sample = New_load.descale(sample, sc)
        scenarios_random.append(sample)

    return np.swapaxes(np.array(scenarios_random), 0, 1)
