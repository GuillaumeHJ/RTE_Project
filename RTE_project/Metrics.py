from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt

import VAE_model
import VAE_model as model
import Load_data
import pandas as pd

path = "data/"

train_dataloader, val_dataloader, test_dataloader, training_set, validation_set, test_set = Load_data.load(path, False)
train_cond_dataloader, val_cond_dataloader, test_cond_dataloader, training_cond, validation_cond, test_cond = Load_data.load(
    path, True)


def histogram_error(vae, conditioned=False):
    err = []
    if conditioned:
        for x in validation_cond:
            decoded_x = vae(x[None, :])
            err.append((((x[:48] - decoded_x[0]) ** 2).sum() / (x.shape[0] - 1)).detach().numpy())
    else:
        for x in validation_set:
            decoded_x = vae(x[None, :])
            err.append(((x - decoded_x) ** 2).mean().detach().numpy())
    plt.hist(err)
    plt.show()
    return err


def plot_year_err(vae, conditioned=False):
    err = []
    if conditioned:
        for x in validation_cond:
            decoded_x = vae(x[None, :])
            err.append((((x[:48] - decoded_x[0]) ** 2).sum() / (x.shape[0] - 1)).detach().numpy())
    else:
        for x in validation_set:
            decoded_x = vae(x[None, :])
            err.append(((x - decoded_x) ** 2).mean().detach().numpy())
    abs = np.arange(365)
    plt.plot(abs, err)
    plt.show()


# hyper parameters
latent_space_dim = 5
hidden_layer_dim = [250, 120, 60]
lr = 1e-4
epochs = 20

vae = VAE_model.CondVAE(latent_space_dim, hidden_layer_dim)
vae.train(epochs, lr)

err = histogram_error(vae, True)
plot_year_err(vae, True)


def make_chronics(df, toshape_columns, pivot_indexcol, pivot_columncol=None):
    """[summary]
    Args:
        df ([type]): [description]
        toshape_columns ([type]): [description]
        pivot_indexcol ([type]): [description]
        pivot_columncol ([type], optional): [description]. Defaults to None.
    Returns:
        [type]: [description]
    """

    assert pivot_indexcol in list(df.columns)
    if pivot_columncol is not None:
        assert pivot_columncol in list(df.columns)

    list_df = []
    print('ok1')
    for col in toshape_columns:
        df_pivot = df[[col, pivot_indexcol, pivot_columncol]].pivot(
            values=col, index=pivot_indexcol, columns=pivot_columncol).copy()
        print('ok2')
        if df_pivot.isna().sum().sum() != 0:
            df_pivot = df_pivot.interpolate(method="linear", axis=0)
        print('ok3')
        list_df.append(df_pivot.copy())

    return tuple(list_df)


def make_df_calendar(df_datetime):
    """[summary]
    Args:
        df_datetime ([type]): [description]
    Returns:
        [type]: [description]
    """

    ds = df_datetime.columns[0]
    df_datetime[ds] = pd.to_datetime(df_datetime[ds])

    df_datetime['month'] = df_datetime[ds].dt.month
    df_datetime['weekday'] = df_datetime[ds].dt.weekday
    df_datetime['is_weekend'] = (df_datetime.weekday >= 5).apply(lambda x: int(x))
    df_datetime['year'] = df_datetime[ds].dt.year

    return df_datetime


def apply_scaler(df, column, df_chronic, reference_window):
    """[summary]
    Args:
        df ([type]): [description]
        column ([type]): [description]
        reference_window ([type]): [description]
    Returns:
        [type]: [description]
    """

    if reference_window is None:
        reference_window = np.array([True] * df.shape[0])

    scaler = StandardScaler().fit(df[[column]].loc[reference_window].values)

    df_chronic = df_chronic.apply(lambda x: scaler.transform(x.reshape(-1, 1)).ravel(), raw=True, axis=1)

    return df_chronic, scaler


# Construction of factorMatrix and factorDesc

# importation des données calendaires

"""
df_data.utc_datetime = pd.to_datetime(df_data.utc_datetime, utc=True)

ds = pd.DataFrame({"days" : df_data.utc_datetime.dt.date, "minute":df_data.utc_datetime.dt.minute+60*df_data.utc_datetime.dt.hour})

df_conso, df_temp, df_prevision = make_chronics(df=pd.concat([df_data, ds], axis=1),
                                               toshape_columns=["Consommation", "prevision_temp", "prevision_j-1"],
                                               pivot_indexcol="days", pivot_columncol="minute")






df_conso, conso_scaler = apply_scaler(df_data, column="Consommation", df_chronic=df_conso,
                                      reference_window=df_data.utc_datetime.dt.year <=2018)



df_calendar = make_df_calendar(pd.DataFrame({"ds" : pd.to_datetime(np.asarray(df_conso.index))}))

df_holidays = pd.concat([df_data[["is_holidays"]],pd.DataFrame({"ds" : pd.to_datetime(ds.days.values)})], axis=1).drop_duplicates(
                                               subset="ds").reset_index(drop= True)

df_calendar = df_calendar.merge(df_holidays, on="ds", how="left").rename(columns={"is_holidays":"is_holiday_day"})



#explicit the potential bridge days taken as extended holidays
day_hol = df_calendar[['weekday', 'is_holiday_day']].copy().values
bridge_index=[]
for i in range(day_hol.shape[0]):
    if day_hol[i,1]==1:
        if day_hol[i,0]==1:
            bridge_index.append(i-1)
        elif day_hol[i,0]==3:
            bridge_index.append(i+1)

bridges = np.zeros(day_hol.shape[0])
bridges[np.asarray(bridge_index)] = 1

df_calendar['potential_bridge_holiday'] = bridges
#calendar_info['potential_bridge_holiday'].describe()

calendar_factors = ["weekday", "is_weekend", "month", "is_holiday_day"]
factors = df_calendar[calendar_factors].copy()
factorDesc = {ff : 'category' for ff in calendar_factors}

temperatureMean= df_temp.mean(axis=1).values.reshape(-1,1)
factorMatrix = np.c_[factors.values,temperatureMean]
factorDesc['temperature']='regressor'

print('factor matrix :' , factorMatrix, 'factor matrix.shape :' , factorMatrix.shape )
print('desc matrix :' , factorDesc)

"""


def disentanglement_quantification(x_reduced, factorMatrix, factorDesc, algorithm='RandomForest', cv=3,
                                   normalize_information=False):
    """criteria based on "A Framework for the Quantitative Evaluation of Disentangled Representations", Eastwood and Williams (2018)
    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of conditions names and types
    factorMatrix -- array-like, array containing conditions values for the representation (columns in the keys order of factorDesc)
    algorithm -- the kind of estimator to make predictions with
    cv -- int, cross-validation generator or an iterable. Determines the cross-validation splitting strategy.
    normalize_information -- Boolean, whether to normalize informativeness results with the minimum obtained with a random projection
    return: final_evaluation -- dict, dict of metrics values
            importance_matrix -- array-like, importance matrix for latent dimensions (rows) to predict factors (columns)
    """
    assert algorithm == 'RandomForest' or algorithm == 'GradientBoosting'
    if algorithm == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier as clf
        from sklearn.ensemble import RandomForestRegressor as reg
    else:
        from sklearn.ensemble import GradientBoostingClassifier as clf
        from sklearn.ensemble import GradientBoostingRegressor as reg

    z_dim = x_reduced.shape[1]
    n_factors = factorMatrix.shape[1]
    evaluation = {}
    evaluation['informativeness'] = {}
    evaluation['importance_variable'] = {}
    final_evaluation = {}
    # estimation of importance of the latent code variables for each factor using random forest attribut of feature importances
    for i, name in enumerate(factorDesc.keys()):
        factor_type = factorDesc[name]
        if (factor_type == 'category'):
            estimator = clf(n_estimators=100)
            cv_results = cross_validate(estimator, x_reduced, factorMatrix[:, i], cv=cv, return_estimator=True,
                                        scoring='f1_macro')
        else:
            estimator = reg(n_estimators=100)
            cv_results = cross_validate(estimator, x_reduced, factorMatrix[:, i], cv=cv, return_estimator=True,
                                        scoring='r2')

        if normalize_information:
            x_reduced_random = np.random.rand(x_reduced.shape[0], 1)
            if (factor_type == 'category'):
                estimator_random = clf(n_estimators=100)
                cv_results_random = cross_validate(estimator_random, x_reduced_random, factorMatrix[:, i], cv=cv,
                                                   return_estimator=True, scoring='f1_macro')
            else:
                estimator_random = reg(n_estimators=100)
                cv_results_random = cross_validate(estimator_random, x_reduced_random, factorMatrix[:, i], cv=cv,
                                                   return_estimator=True, scoring='r2')

        if normalize_information:
            min_info = np.mean(cv_results_random['test_score'])
            results_info = np.nan_to_num((np.mean(cv_results['test_score']) - min_info) / (1 - min_info))
        else:
            results_info = np.mean(cv_results['test_score'])

        evaluation['informativeness'][name] = max(results_info, 0)
        importance_P = np.concatenate([esti.feature_importances_.reshape(-1, 1) for esti in cv_results['estimator']],
                                      axis=1)
        evaluation['importance_variable'][name] = np.mean(importance_P, axis=1)

    final_evaluation['informativeness'] = np.asarray(
        [evaluation['informativeness'][name] for name in factorDesc.keys()])

    importance_matrix = np.concatenate(
        [evaluation['importance_variable'][name].reshape(-1, 1) for name in factorDesc.keys()], axis=1)
    importance_matrix_norm = np.apply_along_axis(lambda x: x / np.sum(x), 1, importance_matrix)

    disentangled_measures = 1 + np.sum(
        importance_matrix_norm * np.log(importance_matrix_norm + 1e-10),
        axis=1) / np.log(n_factors)

    compactness_measures = 1 + np.sum(importance_matrix * np.log(importance_matrix + 1e-10), axis=0) / np.log(z_dim)

    weights_predictonefactor = np.sum(importance_matrix, axis=1) / np.sum(importance_matrix)
    weighted_disentanglement = np.sum(disentangled_measures * weights_predictonefactor)

    final_evaluation['disentanglement'] = disentangled_measures.ravel()
    final_evaluation['compactness'] = compactness_measures.ravel()
    final_evaluation['mean_disentanglement'] = weighted_disentanglement.ravel()

    return final_evaluation, importance_matrix


def compute_mig(x_reduced, factorMatrix, factorDesc, batch=None):
    """criterion Mutual Information Gap implementation based on "Isolating Sources of Disentanglement in Variational Autoencoders", Chen (2018);
       inspiration from disentanglement_lib of Olivier Bachem.
    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of genertaive/explicative names and types
    factorMatrix -- array-like, array explicative factors values for the representation (columns in the keys order of factorDesc)
    batch -- whether to compute the MIG on a sliced part of the latent representation
    :return: mig -- float, MIG average value across the factors
    """

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression

    if batch is None:
        train_size = x_reduced.shape[0]
    else:
        train_size = batch
    sample_index = np.random.choice(range(train_size), size=train_size, replace=False)
    latent = x_reduced[sample_index, :]
    ys = factorMatrix[sample_index, :]

    m = np.zeros((x_reduced.shape[1], factorMatrix.shape[1]))
    entropy = np.zeros(ys.shape[1])
    for j, name in enumerate(factorDesc.keys()):
        factor_type = factorDesc[name]
        if (factor_type == 'category'):
            m[:, j] = mutual_info_classif(latent, ys[:, j]).T
            entropy[j] = mutual_info_classif(ys[:, j].reshape(-1, 1), ys[:, j]).ravel()
        else:
            m[:, j] = mutual_info_regression(latent, ys[:, j]).T
            entropy[j] = mutual_info_regression(ys[:, j].reshape(-1, 1), ys[:, j]).ravel()

    sorted_m = np.sort(m, axis=0)
    mig = np.divide(sorted_m[-1, :] - sorted_m[-2, :], entropy)

    return mig


def compute_modularity(x_reduced, factorMatrix, factorDesc, batch=None):
    """criterion Modularity based on "Learning Deep Disentangled Embeddings With the F-Statistic Loss", Ridgeway and Mozer (2018);
        inspiration from disentanglement_lib of Olivier Bachem.
    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of generative/explicative factors names and types
    factorMatrix -- array-like, array containing explicative factors values for the representation (columns in the keys order of factorDesc)
    batch -- whether to compute the MIG on a sliced part of the latent representation
    :return: modularity -- float, modularity score for the representation
    """

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression

    if batch is None:
        train_size = x_reduced.shape[0]
    else:
        train_size = batch
    sample_index = np.random.choice(range(train_size), size=train_size, replace=False)
    latent = x_reduced[sample_index, :]
    ys = factorMatrix[sample_index, :]

    m = np.zeros((x_reduced.shape[1], factorMatrix.shape[1]))
    for j, name in enumerate(factorDesc.keys()):
        factor_type = factorDesc[name]
        if (factor_type == 'category'):
            m[:, j] = mutual_info_classif(latent, ys[:, j]).T
        else:
            m[:, j] = mutual_info_regression(latent, ys[:, j]).T

    sorted_m = np.r_[[np.eye(1, m.shape[1], k).ravel() for k in np.argmax(m, axis=1)]]
    t_i = m * sorted_m

    d_i = np.sum(np.square(m - t_i), axis=1) / np.square(np.max(m, axis=1)) / (factorMatrix.shape[1] - 1)

    return 1 - d_i


def evaluate_latent_code(x_reduced, factorMatrix, factorDesc, algorithm='RandomForest', cv=3, orthogonalize=True,
                         normalize_information=False):
    """ function to return a dict of implemented metrics which are informativeness, compactness, disentanglement, MIG and modularity
    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of generative/causal factors names and types
    factorMatrix -- array-like, array containing conditions values for the representation (columns in the keys order of factorDesc)
    algorithm -- the kind of estimator to make predictions with
    cv -- int, cross-validation generator or an iterable. Determines the cross-validation splitting strategy.
    orthogonalize -- Boolean, whether to fix the explicative axes of the representation on the coordinates dimensions
    normalize_information -- Boolean, whether to normalize informtiveness results with the minimum obtained with a random projection
    :return: final_evaluation -- dict, dict of metrics values
             importance_matrix -- array-like, importance matrix for latent dimensions (rows) to predict factors (columns)
    """
    if orthogonalize:
        from sklearn.decomposition import PCA
        ortho_proj = PCA(x_reduced.shape[1])
        x = ortho_proj.fit_transform(x_reduced)
    else:
        x = x_reduced

    final_evaluation, importance_matrix = disentanglement_quantification(x, factorMatrix, factorDesc,
                                                                         algorithm=algorithm, cv=cv,
                                                                         normalize_information=normalize_information)
    final_evaluation['mig'] = compute_mig(x, factorMatrix, factorDesc)
    final_evaluation['modularity'] = compute_modularity(x, factorMatrix, factorDesc)

    return final_evaluation, importance_matrix


def display_evaluation_latent_code(final_evaluation, z_dim, factorDesc):
    """A proposition of barplots to display results of the computed disentanglement metrics
    Args:
        final_evaluation (dict): dict of computed metrics values
        z_dim (int): number of latent dimensions
        factorDesc (dict): dict of generative/explicative factors names and types
    """

    if 'reconstruction_error' in final_evaluation.keys():
        for k, v in final_evaluation['reconstruction_error'].item():
            print(k, ' : ', v)

    fig = plt.figure(dpi=100, figsize=(10, 8))

    plt.subplot(2, 3, 1)
    fig.subplots_adjust(hspace=0.5)
    plt.bar(factorDesc.keys(), final_evaluation['informativeness'])
    plt.xlabel('factors')
    plt.xticks(rotation=75)
    plt.ylim(top=1)
    for index, data in enumerate(final_evaluation['informativeness']):
        plt.text(x=index - 0.5, y=data + 0.01, s="%.2f" % data, fontdict=dict(fontsize=10))
    plt.title('Informativeness score : %.2f' % np.mean(final_evaluation['informativeness']))

    plt.subplot(2, 3, 2)
    plt.bar(np.arange(z_dim) + 1, final_evaluation['disentanglement'])
    plt.xlabel('latent variables')
    plt.title('Disentanglement score : %.2f' % final_evaluation['mean_disentanglement']);

    plt.subplot(2, 3, 3)
    plt.bar(factorDesc.keys(), final_evaluation['compactness'])
    plt.xlabel('factors')
    plt.xticks(rotation=75)
    plt.title('Compactness')
    plt.tight_layout();

    plt.subplot(2, 3, 5)
    plt.bar(np.arange(z_dim) + 1, 1 - final_evaluation['modularity'])
    plt.xlabel('latent variables')
    plt.title('Modularity score : %.2f' % np.mean(1 - final_evaluation['modularity']));

    plt.subplot(2, 3, 6)
    plt.bar(factorDesc.keys(), final_evaluation['mig'])
    plt.xlabel('factors')
    plt.xticks(rotation=75)
    plt.title('Mutual Information Gap (MIG) : %.2f' % np.mean(final_evaluation['mig']))
    plt.tight_layout();

    plt.show()


"""
#Metrics

final_evaluation, importance_matrix = evaluate_latent_code(x_reduced, factorMatrix, factorDesc, algorithm='RandomForest', cv=3, orthogonalize=True, normalize_information=False)

display_evaluation_latent_code(final_evaluation, latent_space_dim, factorDesc)

"""
