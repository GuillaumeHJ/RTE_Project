import matplotlib.pyplot as plt
import numpy as np

import ClementCVAE


def generate_scenarios(decoder, test_set, M=100):
    scenarios = []
    for _ in range(M):
        sample = np.random.normal(size=(test_set.shape[0], ClementCVAE.latent_dims))
        generated_scenario = decoder(np.concatenate((sample, test_set[:, 48:50]), axis=1))
        scenarios.append(generated_scenario)

    scenarios = np.swapaxes(np.array(scenarios), 0, 1)
    mean = np.mean(scenarios, axis=1)

    # Computing biases for each scenario
    biases = scenarios - mean[:, np.newaxis, :]

    # Sorting scenarios according to their biases
    sorted_indices = np.argsort(biases, axis=1)
    sorted_scenarios = np.take_along_axis(scenarios, sorted_indices, axis=1)

    #print(f"scenarios : {scenarios.shape}")
    #print(f"mean : {mean.shape}")
    #print(f"biases : {biases.shape}")
    #print(f"indices : {sorted_indices.shape}")
    #print(f"sorted scenarios : {sorted_scenarios.shape}")

    return sorted_scenarios[:, M // 4, :], mean, sorted_scenarios[:, 3 * M // 4, :], sorted_scenarios


q3, mean, q1, scenarios = generate_scenarios(ClementCVAE.decoder, ClementCVAE.x_val)

#print(q3.shape)
#print(scenarios.shape)

day = 150
plt.plot(np.arange(48), q1[day, :48], color='green', linestyle='dashed')
plt.plot(np.arange(48), mean[day, :48], color='red', linestyle='dashed')
plt.plot(np.arange(48), q3[day, :48], color='blue', linestyle='dashed')
plt.show()
