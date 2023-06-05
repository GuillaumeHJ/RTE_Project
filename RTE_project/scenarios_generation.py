import matplotlib.pyplot as plt
import numpy as np
import New_load
import VAE_model
from torch.utils.data import DataLoader
import baseline_scenarios_gen as baseline

clement = False
M = 100
latent_space_dim = 3

if clement:
    import scoring_Clement as scoring

    X_train, X_val, sc = scoring.ClementCVAE.x_train, scoring.ClementCVAE.x_val, scoring.ClementCVAE.sc
    q1, median, q3, scenarios = scoring.generate_scenarios(scoring.ClementCVAE.decoder, X_val, M=M)

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

    q1, median, q3, scenarios = scoring.generate_scenarios(vae_cond.decoder, X_val, latent_space_dim=latent_space_dim,
                                                         sc=sc, M=M)

day = 200
plt.plot(np.arange(48), q1[day, :48], color='green', linestyle='dashed', label='Q1')
plt.plot(np.arange(48), median[day, :48], color='red', linestyle='dashed', label='Median')
plt.plot(np.arange(48), q3[day, :48], color='blue', linestyle='dashed', label='Q3')
plt.plot(np.arange(48), New_load.descale(X_val[:, :48], sc)[day], label='Real Profile')
plt.legend()
plt.show()

value_energy_score = scoring.energy_score(X_val[:353, :], scenarios[:353, :, :])
value_variogram_score = scoring.variogram_score(X_val[:353, :], scenarios[:353, :, :])
value_quantile_score = scoring.quantile_score(X_val[:353, :], scenarios[:353, :, :])
print('Energy score', value_energy_score)
print('Variogram score', value_variogram_score)
print('Quantile score', value_quantile_score)
print('////////////////////////////////// AVERAGE ////////////////////////////////////////')


print('Comparaison avec un scénario moyenné sur le training set')

avg_scenario_broadcasted = baseline.avg_scenario(X_train, X_val, sc, M)

value_energy_score_avg = scoring.energy_score(X_val, avg_scenario_broadcasted)
value_variogram_score_avg = scoring.variogram_score(X_val, avg_scenario_broadcasted)
value_quantile_score_avg = scoring.quantile_score(X_val, avg_scenario_broadcasted)

print('energy score', value_energy_score_avg)
print('variogram score avg', value_variogram_score_avg)
print('quantile score avg', value_quantile_score_avg)
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('Rapport Energy score avg', value_energy_score / value_energy_score_avg)
print('Rapport Variogram score avg', value_variogram_score / value_variogram_score_avg)
print('Rapport Quantile score avg', value_quantile_score / value_quantile_score_avg)
print('//////////////////////////////// PCA //////////////////////////////////////////')

scenarios_pca = baseline.pca_scenario(X_train, sc, latent_space_dim, M)

value_energy_score_pca = scoring.energy_score(X_val, scenarios_pca)
value_variogram_score_pca = scoring.variogram_score(X_val, scenarios_pca)
value_quantile_score_pca = scoring.quantile_score(X_val, scenarios_pca)

print('energy score', value_energy_score_pca)
print('variogram score avg', value_variogram_score_pca)
print('quantile score avg', value_quantile_score_pca)
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('Rapport Energy score avg', value_energy_score / value_energy_score_pca)
print('Rapport Variogram score avg', value_variogram_score / value_variogram_score_pca)
print('Rapport Quantile score avg', value_quantile_score / value_quantile_score_pca)
print('//////////////////////////////// RANDOM //////////////////////////////////////////')

scenarios_random = baseline.random_scenario(X_train, X_val, sc, M)

value_energy_score_random = scoring.energy_score(X_val, scenarios_random)
value_variogram_score_random = scoring.variogram_score(X_val, scenarios_random)
value_quantile_score_random = scoring.quantile_score(X_val, scenarios_random)

print('energy score', value_energy_score_random)
print('variogram score avg', value_variogram_score_random)
print('quantile score avg', value_quantile_score_random)
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('Rapport Energy score avg', value_energy_score / value_energy_score_random)
print('Rapport Variogram score avg', value_variogram_score / value_variogram_score_random)
print('Rapport Quantile score avg', value_quantile_score / value_quantile_score_random)
print('//////////////////////////////////////////////////////////////////////////')

