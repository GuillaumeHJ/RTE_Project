import pandas as pd

import ClementCVAE
import numpy as np
from sklearn.metrics import mean_squared_error


def generate_scenarios(decoder, test_set, M=100):
    scenarios = []
    for _ in range(M):
        # Monte Carlo on N(0,1)
        sample = np.random.normal(size=(test_set.shape[0], ClementCVAE.latent_dims))

        # Generating scenarios from the latent space
        generated_scenario = decoder(np.concatenate((sample, test_set[:, 48:50]), axis=1))

        # Rescaling it
        df_scenario = pd.DataFrame(generated_scenario[:, :48])
        df_scenario_descaled = df_scenario.apply(
            lambda x: ClementCVAE.sc.inverse_transform(x.reshape(-1, 1)).ravel(), axis=1, raw=True).values
        scenario_descaled = np.array(list(df_scenario_descaled))
        scenarios.append(np.concatenate((scenario_descaled, generated_scenario[:, 48:50]), axis=1))

    scenarios = np.swapaxes(np.array(scenarios), 0, 1)
    mean = np.mean(scenarios, axis=1)

    # Computing biases for each scenario
    biases = scenarios - mean[:, np.newaxis, :]

    # Sorting scenarios according to their biases
    sorted_indices = np.argsort(biases, axis=1)
    sorted_scenarios = np.take_along_axis(scenarios, sorted_indices, axis=1)

    # print(f"scenarios : {scenarios.shape}")
    # print(f"mean : {mean.shape}")
    # print(f"biases : {biases.shape}")
    # print(f"indices : {sorted_indices.shape}")
    # print(f"sorted scenarios : {sorted_scenarios.shape}")

    return sorted_scenarios[:, M // 4, :], mean, sorted_scenarios[:, 3 * M // 4, :], sorted_scenarios


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
