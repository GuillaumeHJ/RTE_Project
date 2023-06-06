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

    X_train, X_val, sc, weekdays, months, df_prev_val = scoring.ClementCVAE.x_train, scoring.ClementCVAE.x_val, scoring.ClementCVAE.sc, scoring.ClementCVAE.weekdays, scoring.ClementCVAE.months, scoring.ClementCVAE.df_prev_val
    q1, median, q3, scenarios = scoring.generate_scenarios(scoring.ClementCVAE.decoder, X_val, sc=sc, M=M)
    q1_vae, median_vae, q3_vae, scenarios_vae = scoring.generate_scenarios(scoring.ClementVAE.decoder, X_val, sc=sc,
                                                                           M=M, conditioned=False)

else:
    import scoring_pytorch as scoring

    path = "data/"

    X_train, X_val, sc, weekdays, months, df_prev_val = New_load.load(path, days=True, month=True)

    train_dataloader = DataLoader(X_train, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(X_val, batch_size=64, shuffle=True)

    # hyper parameters
    hidden_layer_encoder = [48, 48, 24, 12]
    hidden_layer_decoder = [12, 24]
    lr = 5 * 1e-5
    epochs = 200


    vae_cond = VAE_model.VAE(latent_space_dim, hidden_layer_encoder, hidden_layer_decoder, conditioned=True)
    vae = VAE_model.VAE(latent_space_dim, hidden_layer_encoder, hidden_layer_decoder, conditioned=False)
    l, v = vae_cond.train(epochs, lr, train_dataloader, val_dataloader)
    l, v = vae.train(epochs, lr, train_dataloader, val_dataloader)

    q1, median, q3, scenarios = scoring.generate_scenarios(vae_cond.decoder, X_val, latent_space_dim=latent_space_dim,
                                                           sc=sc, M=M)

    q1_vae, median_vae, q3_vae, scenarios_vae = scoring.generate_scenarios(vae.decoder, X_val,
                                                                           latent_space_dim=latent_space_dim,
                                                                           sc=sc, M=M, conditioned=False)



# Plotting predictions with the CVAE
day = 200
plt.plot(np.arange(48), q1[day, :48], color='green', linestyle='dashed', label='Q1')
plt.plot(np.arange(48), median[day, :48], color='red', linestyle='dashed', label='Median')
plt.plot(np.arange(48), q3[day, :48], color='blue', linestyle='dashed', label='Q3')
plt.plot(np.arange(48), New_load.descale(X_val[:, :48], sc)[day], label='Real Profile')
plt.legend()
plt.show()

# Plotting predictions with the VAE unconditionned
plt.plot(np.arange(48), q1_vae[day, :48], color='green', linestyle='dashed', label='Q1')
plt.plot(np.arange(48), median_vae[day, :48], color='red', linestyle='dashed', label='Median')
plt.plot(np.arange(48), q3_vae[day, :48], color='blue', linestyle='dashed', label='Q3')
plt.plot(np.arange(48), New_load.descale(X_val[:, :48], sc)[day], label='Real Profile')
plt.legend()
plt.show()

print('VALEURS MODELE CVAE')
value_energy_score = scoring.energy_score(New_load.descale(X_val[:353, :49], sc), scenarios[:353, :, :])
value_variogram_score = scoring.variogram_score(New_load.descale(X_val[:353, :49], sc), scenarios[:353, :, :])
value_quantile_score = scoring.quantile_score(New_load.descale(X_val[:353, :49], sc), scenarios[:353, :, :])
value_mse_score = scoring.MSE(New_load.descale(X_val[:, :48], sc), median[:, :48])

print('mse score', value_mse_score)
print('Energy score CVAE', value_energy_score)
print('Variogram score CVAE', value_variogram_score)
print('Quantile score CVAE', value_quantile_score)

print('////////////////////////////////// VAE unconditioned ////////////////////////////////////////')

value_energy_score_vae = scoring.energy_score(New_load.descale(X_val[:353, :49], sc), scenarios_vae[:353, :, :])
value_variogram_score_vae = scoring.variogram_score(New_load.descale(X_val[:353, :49], sc), scenarios_vae[:353, :, :])
value_quantile_score_vae = scoring.quantile_score(New_load.descale(X_val[:353, :49], sc), scenarios_vae[:353, :, :])
value_mse_score_vae = scoring.MSE(New_load.descale(X_val[:, :48], sc), median_vae[:, :48])
print('MSE score VAE', value_mse_score_vae)
print('Energy score VAE', value_energy_score_vae)
print('Variogram score VAE', value_variogram_score_vae)
print('Quantile score VAE', value_quantile_score_vae)
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('Rapport MSE score vae', value_mse_score / value_mse_score_vae)
print('Rapport Energy score vae', value_energy_score / value_energy_score_vae)
print('Rapport Variogram score vae', value_variogram_score / value_variogram_score_vae)
print('Rapport Quantile score vae', value_quantile_score / value_quantile_score_vae)

print('////////////////////////////////// AVERAGE ////////////////////////////////////////')
print('Comparaison avec un scénario moyenné sur le training set')

avg_scenario_broadcasted, avg_scenario_adaptmse = baseline.avg_scenario(X_train, X_val, sc, M)

value_energy_score_avg = scoring.energy_score(New_load.descale(X_val[:, :49], sc), avg_scenario_broadcasted)
value_variogram_score_avg = scoring.variogram_score(New_load.descale(X_val[:, :49], sc), avg_scenario_broadcasted)
value_quantile_score_avg = scoring.quantile_score(New_load.descale(X_val[:, :49], sc), avg_scenario_broadcasted)
value_mse_avg = scoring.MSE(New_load.descale(X_val[:, :48], sc), avg_scenario_adaptmse)

print('MSE avg', value_mse_avg)
print('energy score', value_energy_score_avg)
print('variogram score avg', value_variogram_score_avg)
print('quantile score avg', value_quantile_score_avg)
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('Rapport MSE score avg', value_mse_score / value_mse_avg)
print('Rapport Energy score avg', value_energy_score / value_energy_score_avg)
print('Rapport Variogram score avg', value_variogram_score / value_variogram_score_avg)
print('Rapport Quantile score avg', value_quantile_score / value_quantile_score_avg)

print('//////////////////////////////// PCA //////////////////////////////////////////')

scenarios_pca = baseline.pca_scenario(X_train, sc, latent_space_dim, M)

value_energy_score_pca = scoring.energy_score(New_load.descale(X_val[:, :49], sc), scenarios_pca)
value_variogram_score_pca = scoring.variogram_score(New_load.descale(X_val[:, :49], sc), scenarios_pca)
value_quantile_score_pca = scoring.quantile_score(New_load.descale(X_val[:, :49], sc), scenarios_pca)
value_mse_pca = scoring.MSE(New_load.descale(X_val[:, :48], sc), scenarios_pca[:, 50, :])

print('MSE pca', value_mse_pca)
print('energy score pca', value_energy_score_pca)
print('variogram score pca', value_variogram_score_pca)
print('quantile score pca', value_quantile_score_pca)
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('Rapport MSE score pca', value_mse_score / value_mse_pca)
print('Rapport Energy score pca', value_energy_score / value_energy_score_pca)
print('Rapport Variogram score pca', value_variogram_score / value_variogram_score_pca)
print('Rapport Quantile score pca', value_quantile_score / value_quantile_score_pca)

print('//////////////////////////////// RANDOM //////////////////////////////////////////')

scenarios_random = baseline.random_scenario(X_train, X_val, sc, M)

value_energy_score_random = scoring.energy_score(New_load.descale(X_val[:, :49], sc), scenarios_random)
value_variogram_score_random = scoring.variogram_score(New_load.descale(X_val[:, :49], sc), scenarios_random)
value_quantile_score_random = scoring.quantile_score(New_load.descale(X_val[:, :49], sc), scenarios_random)
value_mse_random = scoring.MSE(New_load.descale(X_val[:, :48], sc), scenarios_random[:, 50, :])

print('MSE score rd', value_mse_random)
print('energy score rd', value_energy_score_random)
print('variogram score rd', value_variogram_score_random)
print('quantile score rd', value_quantile_score_random)
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('Rapport MSE score rd', value_mse_score / value_mse_random)
print('Rapport Energy score rd', value_energy_score / value_energy_score_random)
print('Rapport Variogram score rd', value_variogram_score / value_variogram_score_random)
print('Rapport Quantile score rd', value_quantile_score / value_quantile_score_random)

print('///////////////////////////////// PREVISIONS J-1 /////////////////////////////////////////')


previsions = df_prev_val.to_numpy()[:, np.newaxis, :]
previsions = np.repeat(previsions, M, axis=1)

value_energy_score_prevision = scoring.energy_score(New_load.descale(X_val[:, :49], sc), previsions)
value_variogram_score_prevision = scoring.variogram_score(New_load.descale(X_val[:, :49], sc), previsions)
value_quantile_score_prevision = scoring.quantile_score(New_load.descale(X_val[:, :49], sc), previsions)
value_mse_prevision = scoring.MSE(New_load.descale(X_val[:, :48], sc), previsions[:, 50, :])

print('MSE score prev', value_mse_prevision)
print('energy score prev', value_energy_score_prevision)
print('variogram score prev', value_variogram_score_prevision)
print('quantile score prev', value_quantile_score_prevision)
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('Rapport MSE score prev', value_mse_score / value_mse_prevision)
print('Rapport Energy score prev', value_energy_score / value_energy_score_prevision)
print('Rapport Variogram score prev', value_variogram_score / value_variogram_score_prevision)
print('Rapport Quantile score prev', value_quantile_score / value_quantile_score_prevision)
print('//////////////////////////////////////////////////////////////////////////')

