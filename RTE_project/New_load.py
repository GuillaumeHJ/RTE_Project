import os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

def load(path):
    # Loading and converting data to dataframes
    os.listdir(path)
    dataset_csv = os.path.join(path, "data_conso_2012-2021.parquet.brotli")
    df_data = pd.read_parquet(dataset_csv)
    df_data.utc_datetime = pd.to_datetime(df_data.utc_datetime, utc=True)
    df_data = df_data.replace(np.nan, 0)
    ds = pd.DataFrame({"days": df_data.utc_datetime.dt.date,
                       "minute": df_data.utc_datetime.dt.minute + 60 * df_data.utc_datetime.dt.hour})
    df_data["weekday"] = df_data['utc_datetime'].dt.weekday

    df_conso = pd.concat([df_data, ds], axis=1).pivot(values="Consommation", index="days", columns="minute").copy()
    sc = StandardScaler().fit(df_conso.values.ravel().reshape(-1, 1))
    df_conso_scaled = df_conso.apply(lambda x: sc.transform(x.reshape(-1, 1)).ravel(), axis=1, raw=True).copy()

    #Ajout de la temp
    df_temp = pd.concat([df_data, ds], axis=1).pivot(values="prevision_temp", index="days", columns="minute").copy()
    df_temp['avg_temp'] = df_temp.mean(axis=1)
    df_temp['avg_temp'] = df_temp['avg_temp'].apply(lambda x: x*2)
    df_cond = pd.concat([df_conso_scaled, df_temp['avg_temp']], axis=1)

    #Ajout du weekday
    df_weekday = pd.concat([df_data, ds], axis=1).pivot(values="weekday", index="days", columns="minute").copy()
    df_weekday['weekday'] = df_weekday.mean(axis=1)
    df_cond = pd.concat([df_cond, df_weekday['weekday']], axis=1)


    x_train = df_cond.loc[pd.to_datetime(df_conso.index).year <= 2017].values.astype(np.float32)
    x_val = df_cond.loc[pd.to_datetime(df_conso.index).year == 2018].values.astype(np.float32)

    return x_train, x_val, sc