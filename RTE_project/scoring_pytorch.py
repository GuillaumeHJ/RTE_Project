import New_load
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import mean_squared_error




def generate_scenarios(decoder, test_set, latent_space_dim, sc, M=100, conditioned=True):
    scenarios = []
    for _ in range(M):
        # Monte Carlo on N(0,1)
        sample = np.random.normal(size=(test_set.shape[0], latent_space_dim))
        if conditioned:
            test_set = torch.Tensor(test_set)
            sample = torch.Tensor(sample)
            # Generating scenarios from the latent space
            week_day_one_hot = F.one_hot(test_set[:, 49].long(), num_classes=7)
            cond_sample = torch.cat((sample, test_set[:, 48, None], week_day_one_hot), dim=1)
            generated_scenario = decoder(cond_sample)
        else:
            sample = torch.Tensor(sample)
            generated_scenario = decoder(sample)

        descaled_scenario = New_load.descale(generated_scenario.detach().numpy(), sc)
        scenarios.append(descaled_scenario)

    scenarios = np.swapaxes(np.array(scenarios), 0, 1)
    median = np.median(scenarios, axis=1)

    # Computing biases for each scenario
    biases = scenarios - median[:, np.newaxis, :]

    # Sorting scenarios according to their biases
    sorted_indices = np.argsort(biases, axis=1)
    sorted_scenarios = np.take_along_axis(scenarios, sorted_indices, axis=1)

    # print(f"scenarios : {scenarios.shape}")
    # print(f"mean : {mean.shape}")
    # print(f"biases : {biases.shape}")
    # print(f"indices : {sorted_indices.shape}")
    # print(f"sorted scenarios : {sorted_scenarios.shape}")

    return sorted_scenarios[:, M // 4, :], median, sorted_scenarios[:, 3 * M // 4, :], sorted_scenarios


def energy_score(test_set, scenarios):
    # scenario.shape = (365,M,48)
    M = scenarios.shape[1]  # number of scenarios
    # print('M', M)

    test_set_np = test_set[:, :48]  # (365,48)
    n, p = test_set[:, :48].shape
    # print('n,p', n, p)
    broadcast_test = np.expand_dims(test_set_np, axis=1).repeat(M, axis=1)  # (365,M,48)
    # print('broadcast_test', broadcast_test)
    # print('scenarios', scenarios[353:,:,:])
    norme = np.linalg.norm(broadcast_test - scenarios, axis=2)
    # print('norme', norme)
    energy_1 = 1 / M * np.sum(norme, axis=1)  # (365)
    # print('energy_1', energy_1)

    energy_2 = np.zeros(n)
    for i in range(M):
        for j in range(M):
            energy_2 += np.linalg.norm(scenarios[:, i, :] - scenarios[:, j, :], axis=1)
    energy_2 = 1 / (2 * M ** 2) * energy_2  # (365)
    # print('energy_2', energy_2)

    ES_d = energy_1 - energy_2
    ES = 1 / n * np.sum(ES_d)

    return ES


def variogram_score(test_set, scenarios, weights=np.ones((48, 48)), gamma=0.5):
    M = scenarios.shape[1]  # number of scenarios
    test_set_np = test_set[:, :48]  # (365,48)
    n, p = test_set[:, :48].shape
    VS_d = np.zeros(n)

    for l in range(p):
        for m in range(p):
            esperence_lm = 1 / M * np.sum(np.abs(scenarios[:, :, l] - scenarios[:, :, m]) ** gamma, axis=1)
            VS_d += weights[l, m] * (np.abs(test_set_np[:, l] - test_set_np[:, m]) ** gamma - esperence_lm) ** 2
    VS = 1 / n * np.sum(VS_d)
    return VS


def quantile_score(test_set, scenarios):
    M = scenarios.shape[1]  # number of scenarios
    test_set_np = test_set[:, :48]  # (365,48)
    n, p = test_set[:, :48].shape
    quantiles = np.transpose(np.percentile(scenarios, range(1, 100), axis=1), (1, 0, 2))

    rho_q = np.zeros((n, 99, p))
    for q in range(1, 100):
        b = np.greater(quantiles[:, q-1, :] - test_set_np[:, :], np.zeros((n, p)))
        rho_q[:, q - 1, :] = (1 - 0.01 * q) * (quantiles[:, q - 1, :] - test_set_np[:, :]) * b + 0.01 * q * (
                    test_set_np[:, :] - quantiles[:, q-1, :]) * np.logical_not(b)
        # (365,99,48)

    QS_q = 1 / (n * p) * np.sum(np.sum(rho_q, axis=2), axis=0)  # (99)
    QS = 1 / 99 * np.sum(QS_q)

    return QS
def MSE (test_set, mean_scenarios):
    return mean_squared_error(test_set, mean_scenarios)