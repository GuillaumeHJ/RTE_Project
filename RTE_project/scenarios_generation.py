import matplotlib.pyplot as plt

import Load_data
import VAE_model
import numpy as np

path = "data/"

train_dataloader, val_dataloader, test_dataloader, training_set, validation_set, test_set = Load_data.load(path,
                                                                                                           False)
train_cond_dataloader, val_cond_dataloader, test_cond_dataloader, training_cond, validation_cond, test_cond = Load_data.load(
    path, True)

# hyper parameters
latent_space_dim = 4
hidden_layer_dim = [250, 120, 60]
lr = 1e-4
epochs = 20

vae_cond = VAE_model.CondVAE(latent_space_dim, hidden_layer_dim)
l, v = vae_cond.train(epochs, lr)


def generate_scenarios(trained_model, n=100):
    scenarios = []
    biases = []
    for _ in range(n):
        scenarios.append(trained_model.forward(validation_cond)[200].detach().numpy())
    mean = np.mean(scenarios, axis=0)

    for scenario in scenarios:
        biases.append(np.mean(mean - scenario))

    scenarios_ordered_by_bias = [scenario for _, scenario in sorted(zip(biases, scenarios))]

    return scenarios_ordered_by_bias[n // 4], mean, scenarios_ordered_by_bias[3 * n // 4], scenarios_ordered_by_bias


q3, mean, q1, scenarios = generate_scenarios(vae_cond, 200)

plt.plot(np.arange(48), q1, color='green', linestyle='dashed')
plt.plot(np.arange(48), mean, color='red', linestyle='dashed')
plt.plot(np.arange(48), q3, color='blue', linestyle='dashed')
plt.show()
