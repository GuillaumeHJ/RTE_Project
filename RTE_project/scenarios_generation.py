import matplotlib.pyplot as plt
import numpy as np

import scoring

q3, mean, q1, scenarios = scoring.generate_scenarios(scoring.ClementCVAE.decoder, scoring.ClementCVAE.x_val, M=1000)
print(scenarios.shape)

day = 200
plt.plot(np.arange(48), q1[day, :48], color='green', linestyle='dashed')
plt.plot(np.arange(48), mean[day, :48], color='red', linestyle='dashed')
plt.plot(np.arange(48), q3[day, :48], color='blue', linestyle='dashed')
plt.show()

value_energy_score = scoring.energy_score(scoring.ClementCVAE.x_val[:353,:], scenarios[:353,:,:])
value_variogram_score = scoring.variogram_score(scoring.ClementCVAE.x_val[:353,:], scenarios[:353,:,:])
value_quantile_score = scoring.quantile_score(scoring.ClementCVAE.x_val[:353,:], scenarios[:353,:,:])
print('Energy score', value_energy_score)
print('Variogram score', value_variogram_score)
print('Quantile score', value_quantile_score)

print('Comparaison avec un scénario moyenné sur le training set')

n,p = scoring.ClementCVAE.x_val[:,:48].shape
avg_scenario = np.mean(scoring.ClementCVAE.x_train[:,:48],axis=0)
avg_scenario_broadcasted = np.broadcast_to(avg_scenario, (n, 1000, p))

value_energy_score_avg = scoring.energy_score(scoring.ClementCVAE.x_val, avg_scenario_broadcasted)
value_variogram_score_avg = scoring.variogram_score(scoring.ClementCVAE.x_val, avg_scenario_broadcasted)
value_quantile_score_avg = scoring.quantile_score(scoring.ClementCVAE.x_val, avg_scenario_broadcasted)

print('Energy score avg', value_energy_score_avg)
print('Variogram score avg', value_variogram_score_avg)
print('Quantile score avg', value_quantile_score_avg)
