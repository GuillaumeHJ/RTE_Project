import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import New_load
import VAE_model
from torch.utils.data import DataLoader

clement = False
M = 100
latent_space_dim = 3

if clement:
    import scoring_Clement as scoring

    X_train, X_val, sc = scoring.ClementCVAE.x_train, scoring.ClementCVAE.x_val, scoring.ClementCVAE.sc
    q1, mean, q3, scenarios = scoring.generate_scenarios(scoring.ClementCVAE.decoder, X_val, M=M)

else:
    import scoring_pytorch as scoring

    path = "data/"

    X_train, X_val, sc = New_load.load(path)

    train_dataloader = DataLoader(X_train, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(X_val, batch_size=64, shuffle=True)

    # hyper parameters
    hidden_layer_encoder = [48, 48, 24, 12]
    hidden_layer_decoder = [12, 24]
    lr = 5*1e-5
    epochs = 200

    vae_cond = VAE_model.VAE(latent_space_dim, hidden_layer_encoder, hidden_layer_decoder, conditioned=True)
    l, v = vae_cond.train(epochs, lr, train_dataloader, val_dataloader)

    q1, mean, q3, scenarios = scoring.generate_scenarios(vae_cond.decoder, X_val, latent_space_dim=latent_space_dim,
                                                         sc=sc, M=M)

day = 300
plt.plot(np.arange(48), q1[day, :48], color='green', linestyle='dashed')
plt.plot(np.arange(48), mean[day, :48], color='red', linestyle='dashed')
plt.plot(np.arange(48), q3[day, :48], color='blue', linestyle='dashed')
plt.plot(np.arange(48), New_load.descale(X_val[:, :48], sc)[day])
plt.show()

#Valeurs modèle VAE
print('VALEURS MODELE VAE')
value_energy_score = scoring.energy_score(X_val[:353, :], scenarios[:353, :, :])
value_variogram_score = scoring.variogram_score(X_val[:353, :], scenarios[:353, :, :])
value_quantile_score = scoring.quantile_score(X_val[:353, :], scenarios[:353, :, :])
value_mse_score = scoring.MSE(X_val[:, :48], mean[:, :48])

print('mse score',value_mse_score)
print('Energy score', value_energy_score)
print('Variogram score', value_variogram_score)
print('Quantile score', value_quantile_score)
print('//////////////////////////////////////////////////////////////////////////')


print('Comparaison avec un scénario moyenné sur le training set')

n, p = X_val[:, :48].shape
rescaled_scenarios = New_load.descale(X_val[:, :48], sc)
avg_scenario = np.mean(rescaled_scenarios, axis=0)
avg_scenario_broadcasted = np.broadcast_to(avg_scenario, (n, M, p))
avg_scenario_adaptmse = np.broadcast_to(avg_scenario, (n, p))

value_energy_score_avg = scoring.energy_score(X_val, avg_scenario_broadcasted)
value_variogram_score_avg = scoring.variogram_score(X_val, avg_scenario_broadcasted)
value_quantile_score_avg = scoring.quantile_score(X_val, avg_scenario_broadcasted)
value_mse_avg = scoring.MSE(X_val[:, :48], avg_scenario_adaptmse)

print('mse avg',value_mse_avg)
print('energy score avg', value_energy_score_avg)
print('variogram score avg', value_variogram_score_avg)
print('quantile score avg', value_quantile_score_avg)
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('score modele / score avg')
print('Rapport mse score avg', value_mse_score/value_mse_avg)
print('Rapport Energy score avg', value_energy_score / value_energy_score_avg)
print('Rapport Variogram score avg', value_variogram_score / value_variogram_score_avg)
print('Rapport Quantile score avg', value_quantile_score / value_quantile_score_avg)
print('//////////////////////////////////////////////////////////////////////////')

pca = PCA(n_components=latent_space_dim)
pca.fit(X_train[:, :48])
scenarios_pca = []
for _ in range(M):
    sample = np.random.normal(size=(n, latent_space_dim))
    predict_pca = pca.inverse_transform(sample)
    predict_pca = New_load.descale(predict_pca, sc)
    scenarios_pca.append(predict_pca)

scenarios_pca = np.swapaxes(np.array(scenarios_pca), 0, 1)

value_energy_score_pca = scoring.energy_score(X_val, scenarios_pca)
value_variogram_score_pca = scoring.variogram_score(X_val, scenarios_pca)
value_quantile_score_pca = scoring.quantile_score(X_val, scenarios_pca)
value_mse_pca = scoring.MSE(X_val[:,:48], scenarios_pca[:,50,:])

print('mse pca',value_mse_pca)
print('energy score pca', value_energy_score_pca)
print('variogram score pca', value_variogram_score_pca)
print('quantile score pca', value_quantile_score_pca)
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('score modele / score pca')
print('Rapport mse score pca', value_mse_score/value_mse_pca)
print('Rapport Energy score pca', value_energy_score / value_energy_score_pca)
print('Rapport Variogram score pca', value_variogram_score / value_variogram_score_pca)
print('Rapport Quantile score pca', value_quantile_score / value_quantile_score_pca)

