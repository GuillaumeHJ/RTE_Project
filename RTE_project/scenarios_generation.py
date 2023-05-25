import matplotlib.pyplot as plt
import numpy as np

import ClementCVAE


def generate_scenarios(decoder, test_set, M=100):
    scenarios = []
    for _ in range(M):
        sample = np.random.normal(size=(test_set.shape[0], ClementCVAE.latent_dims))
        generated_scenario = decoder(np.concatenate((sample, test_set[:, 48:50]), axis=1))
        scenarios.append(generated_scenario)

    scenarios = np.array(scenarios)
    mean = np.mean(scenarios, axis=0)
    biases = np.mean(scenarios - mean[None, :, :], axis=2)
    print(biases.shape)

    scenarios_ordered_by_bias = np.sort()

    return scenarios_ordered_by_bias[M // 4], mean, scenarios_ordered_by_bias[3 * M // 4], scenarios_ordered_by_bias

q3, mean, q1, scenarios = generate_scenarios(ClementCVAE.decoder, ClementCVAE.x_val)

day = 100
plt.plot(np.arange(48), q1[day, :48], color='green', linestyle='dashed')
plt.plot(np.arange(48), mean[day, :48], color='red', linestyle='dashed')
plt.plot(np.arange(48), q3[day, :48], color='blue', linestyle='dashed')
plt.show()
