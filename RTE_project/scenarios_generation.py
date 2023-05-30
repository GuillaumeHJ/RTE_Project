import matplotlib.pyplot as plt
import numpy as np

import scoring

q3, mean, q1, scenarios = scoring.generate_scenarios(scoring.ClementCVAE.decoder, scoring.ClementCVAE.x_val, M=100)
print(scenarios.shape)

day = 200
plt.plot(np.arange(48), q1[day, :48], color='green', linestyle='dashed')
plt.plot(np.arange(48), mean[day, :48], color='red', linestyle='dashed')
plt.plot(np.arange(48), q3[day, :48], color='blue', linestyle='dashed')
plt.show()


value_energy_score = scoring.energy_score(scoring.ClementCVAE.decoder, scoring.ClementCVAE.x_val, scenarios[:353, :, :])
value_variogram_score = scoring.variogram_score(scoring.ClementCVAE.decoder, scoring.ClementCVAE.x_val,
                                                scenarios[:353, :, :])

print('Energy score', value_energy_score)
print('Variogram score', value_variogram_score)

mean_scenario = np.mean(scoring.ClementCVAE.x_train[:, :48], axis=0)

