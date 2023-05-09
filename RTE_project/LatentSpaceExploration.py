import Load_data
import VAE_model

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

path = "data/"

train_dataloader, val_dataloader, test_dataloader, training_set, validation_set, test_set = Load_data.load(path,
                                                                                                              False)
train_cond_dataloader, val_cond_dataloader, test_cond_dataloader, training_cond, validation_cond, test_cond = Load_data.load(
    path, True)

# hyper parameters
latent_space_dim = 5
hidden_layer_dim = [250, 120, 60]
lr = 1e-4
epochs = 20

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
