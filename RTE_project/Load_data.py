import sys
import os
import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch


def load(path, conditionned=True):
    # Loading and converting data to dataframes
    os.listdir(path)
    dataset_csv = os.path.join(path, "data_conso_2012-2021.parquet.brotli")
    df_data = pd.read_parquet(dataset_csv)

    # Converting dataframes to normalized tensors of shape (N, 48)
    N = len(df_data)
    df_data = df_data.replace(np.nan, 0)
    dataset = torch.Tensor(df_data['Consommation'].values)
    dataset = dataset[:N - 2 * 365 * 48 - 48]  # without pandemic years

    mean = torch.mean(dataset)
    std = torch.std(dataset)

    dataset = (dataset - mean) / std  # fct nom scikit learn

    training_set = dataset[:N - 4 * 365 * 48 - 48]  # 2012-2017
    training_set = training_set.view((6 * 365 + 2, 48))

    validation_set = dataset[N - 4 * 365 * 48 - 48:N - 3 * 365 * 48 - 48]  # 2018
    validation_set = validation_set.view((365, 48))

    test_set = dataset[N - 3 * 365 * 48 - 48:N - 2 * 365 * 48 - 48]  # 2019
    test_set = test_set.view((365, 48))

    # Plotting average scenarios
    plt.plot(np.arange(48), training_set.mean(axis=0), label='training')
    plt.plot(np.arange(48), validation_set.mean(axis=0), label='validation')
    plt.plot(np.arange(48), test_set.mean(axis=0), label='test')
    plt.legend()

    # Building same tensors with the average temperature
    dataset_temp = torch.Tensor(df_data['prevision_temp'].values)
    dataset_temp = dataset_temp[:N - 2 * 365 * 48 - 48]  # without pandemic years

    mean_temp = torch.mean(dataset_temp)
    std_temp = torch.std(dataset_temp)

    dataset_temp = (dataset_temp - mean_temp) / std_temp  # fct nom scikit learn

    training_set_temp = dataset_temp[:N - 4 * 365 * 48 - 48]  # 2012-2017
    training_set_temp = training_set_temp.view((6 * 365 + 2, 48)).mean(axis=1) * 2

    validation_set_temp = dataset_temp[N - 4 * 365 * 48 - 48:N - 3 * 365 * 48 - 48]  # 2018
    validation_set_temp = validation_set_temp.view((365, 48)).mean(axis=1) * 2

    test_set_temp = dataset_temp[N - 3 * 365 * 48 - 48:N - 2 * 365 * 48 - 48]  # 2019
    test_set_temp = test_set_temp.view((365, 48)).mean(axis=1) * 2

    training_cond = torch.cat((training_set, training_set_temp[:, None]), dim=1)
    validation_cond = torch.cat((validation_set, validation_set_temp[:, None]), dim=1)
    test_cond = torch.cat((test_set, test_set_temp[:, None]), dim=1)

    train_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(validation_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

    train_cond_dataloader = DataLoader(training_cond, batch_size=64, shuffle=True)
    val_cond_dataloader = DataLoader(validation_cond, batch_size=64, shuffle=True)
    test_cond_dataloader = DataLoader(test_cond, batch_size=64, shuffle=True)

    if conditionned:
        return train_cond_dataloader, val_cond_dataloader, test_cond_dataloader, training_cond, validation_cond, test_cond
    else:
        return train_dataloader, val_dataloader, test_dataloader, training_set, validation_set, test_set
