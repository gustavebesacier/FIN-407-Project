import pandas as pd
import numpy as np
import seaborn as sns
import csv
import os

import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from itertools import product
from tqdm import tqdm



RETURN_COLUMN = "STreversal"
LAG_RETURN_COLUMN = "return_lag"
MAX_ITER = 1000000
RANDOM_STATE = 42

L1_RATIO = [0.05, 0.1, 0.5, .95]
METHODS = ['lasso', 'elastic']
DATES = ['198501_200001', '200501_202001', '199501_201001']
FOLDERS = ['output', 'output/figures', '../Predicting/figures', '../Predicting/FAST_weights']


def clean_semicolumns(data):
    """Removes columns containing formatting issues, mainly due to circular savings and generation of csv files.  

    Args:
        data (dataframe): dataframe to be cleaned.

    Returns:
        data: cleaned data
    """
    for name in data.columns.tolist():
        if ":" in name:
            data = data.drop([name], axis =1)
    return data


def order_columns(data, already_lagged=True):
    """Orders the columns from the dataframe so the first 2 would be "permno", "yyyymm" and the 2 last "{RETURN_COLUMN}", "{LAG_RETURN_COLUMN}".
    If the opt

    Args:
        data (dataframe): dataframe to sort. 
        already_lagged (bool): default = True. If not set to False, it ignores the {LAG_RETURN_COLUMN} column.

    Returns:
        data: sorted dataframe
    """
    if already_lagged:
        col = [pred for pred in data.columns.tolist() if not pred in ["permno", "yyyymm", RETURN_COLUMN, LAG_RETURN_COLUMN]]
        data = data[["permno", "yyyymm"] + col + [RETURN_COLUMN, LAG_RETURN_COLUMN]]
    else:
        col = [pred for pred in data.columns.tolist() if not pred in ["permno", "yyyymm", RETURN_COLUMN]]
        data = data[["permno", "yyyymm"] + col + [RETURN_COLUMN]]
    
    return data


def prepare_data(data, dep_variable= None, order_data = False):
    """Returns the vector of dependent variable and independent variables.

    Args:
        data (dataframe): dataframe of returns and predictors. Format: [permno, yyyymm, [predictors], return]

    Returns:
        X: independent variables (predictors)
        y: dependent variable    (returns)
        name_predictors: list of the predictors' name
    """
    if order_data:
        data = clean_semicolumns(order_columns(data))
    else: 
        data = clean_semicolumns(data)

    name_columns = data.columns.tolist()
    
    # name_predictors = name_columns[2:-1]
    # name_predictors = [pred for pred in name_columns if not pred in ['yyyymm', 'permno', LAG_RETURN_COLUMN]]
    # X = data[name_predictors].values
    if dep_variable:
        name_predictors = [pred for pred in name_columns if not pred in ['yyyymm', 'permno', LAG_RETURN_COLUMN, dep_variable]]
        X = data[name_predictors].values
        y = data[dep_variable].values
    else:
        name_predictors = [pred for pred in name_columns if not pred in ['yyyymm', 'permno', LAG_RETURN_COLUMN]]
        X = data[name_predictors].values
        y = data[LAG_RETURN_COLUMN].values

    return X, y, name_predictors


def scale_all_data(X, y):
    """Scale the data.

    Args:
        X (matrix): features matrix.
        y (vector, optional): Dependent variable. Defaults to None.

    Returns:
        X_std: scaled matrix of features.
        y_std: scaled vector of dependent variables.
        scaler: scaler fitted on the matrix of features.
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    y_std = (y - np.mean(y)) / np.var(y)

    return X_std, y_std, scaler


def prepare_lagged_data(data):
    f"""Add the column for lagged return, to perform analysis.
    Create a column {LAG_RETURN_COLUMN} as being the lag 1 period lag of column {RETURN_COLUMN}.

    Args:
        data (dataframe): set of predictors.

    Returns:
        data_lagged: data with lagged return column. 
    """
    
    # Sort the columns
    data = order_columns(data)

    # Sort the values
    data.sort_values('permno', inplace=True)

    # Create the first set of lagged returns for the first permno
    data_lagged = data[data["permno"] == data["permno"].unique()[0]].sort_values(by = ['yyyymm'])
    data_lagged[LAG_RETURN_COLUMN] = data_lagged[RETURN_COLUMN].shift(-1)
    data_lagged.dropna(inplace=True)

    # Avoid any issues by checking the format is fine
    data_lagged = clean_semicolumns(data_lagged)

    # Concatenate dataframe for each permno
    for permno in tqdm(data.permno.unique()):
        d = data[data["permno"] == permno].sort_values(by = ['yyyymm'])
        d[LAG_RETURN_COLUMN] = d[RETURN_COLUMN].shift(-1)               # the predictors of time t are used to forecast return of time t+1
        d.dropna(inplace=True)                                          # we loose 1 observation per company (shift)
        data_lagged = pd.concat([data_lagged, d], ignore_index=True, join='outer')    # add the dataframe of this permno to all the others                                           # takes time to run ~6min

    return data_lagged

    
def return_name(name_predictors, list_selected):
    """Transform list of index of predictors to list of name predictors.

    Args:
        name_predictors (list): list of the name of all predictors of the data (ex:['AM', 'AbnormalAccruals', 'Accruals'])
        list_selected (list): list of indexes of the selected predictors.

    Returns:
        list_cleared: list of the names of the selected predictors. 

    Example:
    >>> predictors = ['AM', 'AOP', 'AbnormalAccruals', 'Accruals', 'AccrualsBM', 'Activism1', 'Activism2', 'AdExp', 'AgeIPO']
    >>> selected_predictors = [0, 2, 3, 4, 6, 8]
    >>> print(return_name(predictors, selected_predictors))
    ['AM', 'AbnormalAccruals', 'Accruals', 'AccrualsBM', 'Activism2', 'AgeIPO']
    """
    list_cleared = [name_predictors[list_selected[i]] for i in range(len(list_selected))]
    return list_cleared


def plot_density_return(data_list):
    _, axs = plt.subplots(1, 3, figsize=(25, 7), sharey=False)

    for i, date in enumerate(data_list):
        print(i)
        date_file = date.split(".")[0].split("/")[1].split("_")
        year_start, year_end = date_file[-2][0:4], date_file[-1][0:4]
        X, y, _ = prepare_data(pd.read_csv(date))
        _, y_std, _ = scale_all_data(X, y)
        x = np.arange(min(y_std), max(y_std), 0.01)

        axs[i].hist(y_std, bins = 200, density=True, color = "black", alpha = 0.5)
        axs[i].plot(x, norm.pdf(x, np.mean(y_std), np.std(y_std)), color = "red")
        axs[i].set_title(f"Lagged return distribution for period {year_start} to {year_end}")
        axs[i].set_xlabel("Standardized lagged return")
        axs[i].set_ylabel("Annualized return")

    plt.savefig("output/figures/distribution_lagged_returns")
    plt.show()

def create_folders():
    for folder in FOLDERS:
        if not os.path.exists(folder):
            os.makedirs(folder)


def bootstrap_result_to_dic(bootstrap_res, name_predictors):
    """Transform list frequency of use of the predictors after bootstrap in dict.

    Args:
        bootstrap_res (list): list of the frequency of each predictors in the bootstrap
        name_predictors (list): list of the name of all predictors of the data (ex:['AM', 'AbnormalAccruals', 'Accruals'])

    Returns:
        dict_boot: dict of {name_predictor: relative_frequency} 

    Example:
    >>> predictors = ['AM', 'AOP', 'AbnormalAccruals', 'Accruals', 'AccrualsBM', 'Activism1', 'Activism2', 'AdExp', 'AgeIPO']
    >>> results = [1.  , 0.09, 1.  , 0.76, 0.98, 0.39, 0.68, 0.  , 0.99]
    >>> print(bootstrap_result_to_dic(results, predictors))
    {'AM': 1.0, 'AOP': 0.09, 'AbnormalAccruals': 1.0, 'Accruals': 0.76, 'AccrualsBM': 0.98, 'Activism1': 0.39, 'Activism2': 0.68, 'AdExp': 0.0, 'AgeIPO': 0.99}
    """

    dict_boot = {name_predictors[i]:bootstrap_res[i] for i in range(len(name_predictors))}
    return dict_boot


def disp_bootstrap_weights(weights, percent = True):
    """Displays the weights from a dictionnary"""
    if percent:
        for (k,v) in enumerate(weights):
            print(f"- {v}: {round(weights[v]*100, 4)}%")

    else:
        for (k,v) in enumerate(weights):
            print(f"- {v}: {round(weights[v], 4)}")


def predictor_selection(weights, tol=0.20):
    """Based on the dict of the weights form a regression, returns the name of predictors with frequency larger than threshold"""
    
    final_predictors = []
    
    for pred, coef in weights.items():
        if coef > tol:
            final_predictors.append(pred)
    
    return final_predictors


def save_list_preds_csv(list_predictors, file_path):
    """Takes the list of predictors and a file path to save the list in a csv file. 

    Args:
        list_predictors (list): list of predictors. 
        file_path (str): file destination. 
    """
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list_predictors)


def print_details_training(dic_results, predictor_name, selected_predictors_opt, lambda_opt=None):
    """Used to display information about the weights and the frequency of each predictors during training. Only if verbose is True in bootstrapping.

    Args:
        dic_results (dict): dictionnary of the results.
        lambda_opt (float): optimal value of lambda.
        predictor_name (list): list of the predictors' names.
        selected_predictors_opt (list): selected features.
    """
    if lambda_opt:
        print("Optimal lambda value: ", lambda_opt)
    # print("Predictor names: ", return_name(predictor_name, selected_predictors_opt))
    print("\nWeights : ")
    disp_bootstrap_weights(dic_results, percent=False)


def cross_validation_LASSO(X, y, lmbdas, nb_kfolds=10, random_state= RANDOM_STATE, verbose = 0, max_iter = MAX_ITER, return_scaler=False, n_jobs = -1):
    """Cross validation using LASSO model.

    Args:
        X (_type_):Feature matrix.
        y (_type_): Target_vector (lagged_return).
        lmbdas (list): List of lambda values of the grid search.
        nb_kfolds (int, optional): number of sets of the cross validation. Defaults to 10.
        random_state (int, optional): random state. Defaults to RANDOM_STATE.
        verbose (int, optional): Level of textual description. 3 for max, 2, 1, 0. Defaults to 0.
        max_iter (int, optional): maximum number of iterations. Defaults to MAX_ITER.
        return_scaler (bool, optional): if True, the function returns the scaler object, fitted on X. Defaults to False.
        n_jobs (int, optional): number of core used. Defaults to -1.

    Returns:
        lambda_opt: optimal value of the penalization coefficient.
        selected_predictors_opt: list of indices of the non-zero coefficients.
        coefficients: value of the coefficients with the optimal penalization coefficient.
        scaler (optional): scaler object, fitted on X
    """

    # Features need to be scaled
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X) # only tranforms features, not the returns
    
    # Initialize the model, then fit it to the scaled data.
    model = LassoCV(alphas = lmbdas, cv = nb_kfolds,  max_iter=max_iter, n_jobs = n_jobs, verbose = verbose, random_state=random_state)
    model.fit(X_std, y)

    # Get the results
    lambda_opt = model.alpha_       # optimal penalization coefficient
    coefficients = model.coef_      # value to the coefficients
    selected_predictors_opt = np.where(coefficients!= 0)[0] # index of optimal coefs

    if return_scaler:
        return lambda_opt, selected_predictors_opt, coefficients, scaler
    else:
        return lambda_opt, selected_predictors_opt, coefficients


def cross_validation_ELASTIC(X, y, lmbdas, nb_kfolds=10, l1_ratio=L1_RATIO, random_state= RANDOM_STATE, verbose = 0, max_iter = MAX_ITER, return_scaler=False, n_jobs = -1):
    """Cross validation using Elastic model.

    Args:
        X (_type_):Feature matrix.
        y (_type_): Target_vector (lagged_return).
        lmbdas (list): List of lambda values of the grid search.
        nb_kfolds (int, optional): number of sets of the cross validation. Defaults to 10.
        l1_ratio (list): scaling between LASSO and Ridge penalities. If 1: l1 penalty.
        random_state (int, optional): random state. Defaults to RANDOM_STATE.
        verbose (int, optional): Level of textual description. 3 for max, 2, 1, 0. Defaults to 0.
        max_iter (int, optional): maximum number of iterations. Defaults to MAX_ITER.
        return_scaler (bool, optional): if True, the function returns the scaler object, fitted on X. Defaults to False.
        n_jobs (int, optional): number of core used. Defaults to -1.

    Returns:
        lambda_opt: optimal value of the penalization coefficient.
        selected_predictors_opt: list of indices of the non-zero coefficients.
        coefficients: value of the coefficients with the optimal penalization coefficient.
        scaler (optional): scaler object, fitted on X
    """

    # Features need to be scaled
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X) # only tranforms features, not the returns
    
    # Initialize the model, then fit it to the scaled data.
    model = ElasticNetCV(alphas = lmbdas, 
                         l1_ratio=l1_ratio,
                         cv = nb_kfolds, 
                         max_iter=max_iter, 
                         n_jobs = n_jobs, 
                         verbose = verbose, 
                         random_state=random_state)
    model.fit(X_std, y)

    # Get the results
    lambda_opt = model.alpha_       # optimal penalization coefficient
    l1_opt = model.l1_ratio_        # ratio between 
    coefficients = model.coef_      # value to the coefficients
    selected_predictors_opt = np.where(coefficients!= 0)[0] # index of optimal coefs

    if return_scaler:
        return lambda_opt, selected_predictors_opt, coefficients, l1_opt, scaler
    else:
        return lambda_opt, selected_predictors_opt, coefficients, l1_opt


def bootstrap_LASSO(X, y, lmbdas, n_bootstraps=100, nb_kfolds=5, verbose=False, max_iter = MAX_ITER, n_jobs = -1, random_state = RANDOM_STATE):
    """
    Perform bootstrap enhanced LASSO.
    
    Parameters:
    - X: Feature matrix.
    - y: Target vector.
    - lambda_values: List of lambda values to try.
    - n_bootstraps: Number of bootstrap samples.
    - k_folds: Number of folds for cross-validation.
    
    Returns:
    - bootstrap_results: frequency of each variable being selected.
    - coef_boot: average value of the coefficients.
    """

    n_samples = X.shape[0]
    n_features = X.shape[1]
    bootstrap_results = np.zeros(n_features)
    coef_boot = [0 for _ in range(X.shape[1])]
    
    for i in tqdm(range(n_bootstraps)):

        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        
        lmbda_opti, selected_predictors, coefficients = cross_validation_LASSO(X = X_bootstrap, 
                                                                                   y = y_bootstrap, 
                                                                                   lmbdas = lmbdas, 
                                                                                   nb_kfolds = nb_kfolds, 
                                                                                   random_state = random_state,
                                                                                   verbose = False, 
                                                                                   max_iter = max_iter, 
                                                                                   return_scaler=False, 
                                                                                   n_jobs = n_jobs)
        # if verbose:
        #     print(f"Iteration number {i+1}. Optimal lambda is {lmbda_opti}")
        coef_boot = list(map(lambda x, y: x + y, coef_boot, coefficients)) # sum all the coefficients at each step, to take average

        # Update the frequency of each variable being selected
        bootstrap_results[selected_predictors] += 1
    
    # Normalize the frequencies
    bootstrap_results /= n_bootstraps
    coef_boot = list(map(lambda x: x/len(coef_boot), coef_boot))
    
    return bootstrap_results, coef_boot


def bootstrap_ELASTIC(X, y, lmbdas, l1_ratio=L1_RATIO, n_bootstraps=100, nb_kfolds=5, verbose=False, max_iter = MAX_ITER, n_jobs = -1, random_state = RANDOM_STATE):
    """
    Perform bootstrap enhanced Elastic Net.
    
    Parameters:
    - X: Feature matrix.
    - y: Target vector.
    - lambda_values: List of lambda values to try.
    - l1_ratio (list): scaling between LASSO and Ridge penalities. If 1: l1 penalty.
    - n_bootstraps: Number of bootstrap samples.
    - k_folds: Number of folds for cross-validation.
    
    Returns:
    - bootstrap_results: frequency of each variable being selected.
    - coef_boot: average value of the coefficients.
    """

    n_samples = X.shape[0]
    n_features = X.shape[1]
    bootstrap_results = np.zeros(n_features)
    coef_boot = [0 for _ in range(X.shape[1])]
    
    for i in tqdm(range(n_bootstraps)):

        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        
        lmbda_opti, selected_predictors, coefficients, l1_opt = cross_validation_ELASTIC(X = X_bootstrap, y = y_bootstrap, lmbdas = lmbdas, l1_ratio=l1_ratio, nb_kfolds = nb_kfolds, random_state = random_state, verbose = False, max_iter = max_iter, return_scaler=False, n_jobs = n_jobs)
        if verbose:
            print(f"Iteration number {i+1}. Optimal lambda is {lmbda_opti}")
        coef_boot = list(map(lambda x, y: x + y, coef_boot, coefficients)) # sum all the coefficients at each step, to take average

        # Update the frequency of each variable being selected
        bootstrap_results[selected_predictors] += 1
    
    # Normalize the frequencies
    bootstrap_results /= n_bootstraps
    coef_boot = list(map(lambda x: x/len(coef_boot), coef_boot))
    
    return bootstrap_results, coef_boot, l1_opt


def run_model(data_path = "../Data/final_data_198501_200001.csv", already_lagged = True, tolerance = 0.5, n_jobs = -1,
              cv_lasso = None, boot_lasso = None, lambda_values = None, nb_kfolds_lasso = 10, nb_boot_lasso = 100, 
              cv_elastic = None, l1_ratio=L1_RATIO, boot_elastic = None, nb_boot_elas = None, nb_kfolds_elas = None, 
              drop_na = None, verbose = None, random_state = RANDOM_STATE, max_iter = MAX_ITER):
    """Function that runs the models. Can be modulated so it only runs specific combinations of models (LASSO/Enet, with or w/o bootstrapping)"""
    
    data = pd.read_csv(data_path)

    data = order_columns(data)

    date = data_path[3:].split(".")[0].split("/")[1].split("_")
    year_start, year_end = date[-2], date[-1]

    if drop_na:
        data.dropna(inplace=True)

    if not already_lagged:
        data_lagged = prepare_lagged_data(data) # Add the lag return
    else:
        data_lagged = data

    # Prepare the data: isolate the predictors and the return
    X, y, predictor_name = prepare_data(data_lagged)
    if verbose:
        print("\nData loaded.")

    y = y.ravel() # Adjust so it has correct format for the models

    # Range of lambda values that will be tested (very small here as data is quite correlated)
    if lambda_values is None:
        lambda_values = [x/500 for x in range(1,30)]

    if cv_lasso:
        print(f"\n------------------------ CROSS-VALIDATION LASSO - PERIOD {year_start} - {year_end} ------------------------")

        lambda_lasso_cv, selected_predictors_lasso_cv, coefficients_lasso_cv = cross_validation_LASSO(X, 
                                                                                   y, 
                                                                                   lambda_values, 
                                                                                   nb_kfolds=10, 
                                                                                   random_state= random_state,
                                                                                   verbose = 0, 
                                                                                   max_iter = max_iter, 
                                                                                   return_scaler=False, 
                                                                                   n_jobs = n_jobs)
        
        # cross_validation_LASSO(X, y, alphas=lambda_values, nb_kfolds=nb_kfolds_lasso, random_state = random_state)
        dic_results_lasso_cv = bootstrap_result_to_dic(coefficients_lasso_cv, predictor_name)
        if verbose:
            print_details_training(dic_results_lasso_cv, lambda_lasso_cv, predictor_name, selected_predictors_lasso_cv)

        final_pred_lasso_cv = predictor_selection(dic_results_lasso_cv, tol = tolerance)
        
        save_list_preds_csv(final_pred_lasso_cv, f"output/list_predictors_lasso_cv_{year_start}_{year_end}.csv")
        print(f"Selection of predictors ({len(final_pred_lasso_cv)} kept from {len(dic_results_lasso_cv)})\n", final_pred_lasso_cv)

    if boot_lasso:
        # Run the bootstrapping, as in https://doi.org/10.1016/j.neuroimage.2010.12.028
        print(f"\n-------------------------- BOOTSTRAPPED LASSO - PERIOD {year_start} - {year_end} --------------------------")

        results_lasso_boot, coeffs_lasso_boot = bootstrap_LASSO(X = X, 
                                                      y = y, 
                                                      lmbdas = lambda_values, 
                                                      n_bootstraps=nb_boot_lasso, 
                                                      nb_kfolds=nb_kfolds_lasso, 
                                                      verbose = verbose,
                                                      n_jobs = n_jobs,
                                                      random_state=random_state,
                                                      max_iter=max_iter)

        dic_results_lasso_boot = bootstrap_result_to_dic(results_lasso_boot, predictor_name)
        
        if verbose:
            print_details_training(dic_results_lasso_boot, predictor_name, coeffs_lasso_boot) # double check

        final_pred_lasso_boot = predictor_selection(dic_results_lasso_boot, tol=tolerance)
        save_list_preds_csv(final_pred_lasso_boot, f"output/list_predictors_lasso_bootstrap_{year_start}_{year_end}.csv")
        print(f"Selection of predictors ({len(final_pred_lasso_boot)} kept from {len(coeffs_lasso_boot)})\n", final_pred_lasso_boot)
    
    if cv_elastic:
        print(f"\n------------------------ CROSS-VALIDATION ELASTIC - PERIOD {year_start} - {year_end} ------------------------")

        lambda_elastic_cv, selected_predictors_elastic_cv, coefficients_elastic_cv, _ = cross_validation_ELASTIC(X = X, 
                                                                                     y = y, 
                                                                                     lmbdas = lambda_values, 
                                                                                     l1_ratio = l1_ratio,
                                                                                     nb_kfolds = nb_kfolds_elas, 
                                                                                     random_state = random_state,
                                                                                     verbose = 0, 
                                                                                     max_iter = max_iter, 
                                                                                     return_scaler = False, 
                                                                                     n_jobs = n_jobs)
        
        dic_results_elastic_cv = bootstrap_result_to_dic(coefficients_elastic_cv, predictor_name)
        
        if verbose:
            print_details_training(dic_results_elastic_cv, lambda_elastic_cv, predictor_name, selected_predictors_elastic_cv)
        
        final_pred_elastic_cv = predictor_selection(dic_results_elastic_cv, tol = tolerance)
        save_list_preds_csv(final_pred_elastic_cv, f"output/list_predictors_elastic_cv_{year_start}_{year_end}.csv")
        print(f"Selection of predictors ({len(final_pred_elastic_cv)} kept from {len(dic_results_elastic_cv)})\n", final_pred_elastic_cv)

    if boot_elastic:
        # Run the bootstrapping, as in https://doi.org/10.1016/j.neuroimage.2010.12.028
        print(f"\n-------------------------- BOOTSTRAPPED ELASTIC - PERIOD {year_start} - {year_end} --------------------------")

        results_elastic_boot, coeffs_elastic_boot, _ = bootstrap_ELASTIC(X = X, 
                                                        y = y, 
                                                        lmbdas = lambda_values, 
                                                        l1_ratio=l1_ratio,
                                                        n_bootstraps=nb_boot_elas, 
                                                        nb_kfolds=nb_kfolds_elas, 
                                                        verbose = verbose,
                                                        max_iter=max_iter,
                                                        n_jobs=n_jobs)

        dic_results_elastic_boot = bootstrap_result_to_dic(results_elastic_boot, predictor_name)
        
        if verbose:
            print_details_training(dic_results_elastic_boot, predictor_name, coeffs_elastic_boot)

        final_pred_elastic_boot = predictor_selection(dic_results_elastic_boot, tol=tolerance)
        save_list_preds_csv(final_pred_elastic_boot, f"output/list_predictors_lasso_bootstrap_{year_start}_{year_end}.csv")
        print(f"Selection of predictors ({len(final_pred_elastic_boot)} kept from {len(coeffs_elastic_boot)})\n", final_pred_elastic_boot)

#################################### ADDING A DIFFERENT METHODOLOGY ####################################

def degrees_of_freedom(model, X, y, return_all = False):
    """Computes the degrees of freedom of a model (Efron) as being:
    ``df(g) = (1/\sigma^2) * \sum_{i=1}^n Cov(g_i(y_i), y_i)``"""

    sigma_squared = np.var(y)
    g_y = model.predict(X)
    cov_y_pred = np.cov(y, g_y)

    df_g = np.trace(cov_y_pred) / sigma_squared

    if return_all:
        return sigma_squared, g_y, cov_y_pred, df_g
    else:
        return df_g


def risk_function(model, X, y, weight = "aic"):
    """Compute the risk function of the model (objective function for determining optimal hyperparameters).

    Args:
        model (LASSO or Enet): model of interest.
        X: set of features.
        y: set of dependent variables.
        weight (str, optional): weight for the function, either 'aic' or 'bic'. Defaults to 'aic'.

    Returns:
        float: value of the risk function.
    """
    n = len(y) # number of observations

    # determine the weight metric to use
    if weight.lower() == "aic":
        w = 2
    elif weight.lower() == "bic": 
        w = np.log(2)
    else:
        print(f"Selected weight {weight!r} incorrect, should be either 'aic' or 'bic'. 'aic' has been used.")
        w = 2

    sigma_squared = np.var(y) # variance of the dependent variable

    error = mean_squared_error(y, model.predict(X)) # MSE

    return error/(n * sigma_squared) + w * degrees_of_freedom(model, X, y, return_all=False) / n


def finding_best_parameter_LASSO(X, y, lambda_values, weight = "aic", return_errors = False, verbose = False, no_progress = True, max_iter = MAX_ITER, random_state = RANDOM_STATE):
    """Given a set of parameters, compute the value of the risk function for each of them and report the optimal penalization parameter.

    Args:
        X: set of features.
        y: set of dependent variables.
        lambda_values (list): list of penalization parameters to test.
        weight (str, optional): weight for the function, either 'aic' or 'bic'. Defaults to 'aic'.
        return_errors (bool, optional): if True, returns the list of the errors for each tested parameter. Defaults to False.
        verbose (bool, optional): if True, adds some information during training. Defaults to False.
        no_progress (bool, optional): if True, does not display the progress bar. Defaults to True.
        max_iter (int, optional): maximum number of iterations of the LASSO function. Defaults to MAX_ITER.
        random_state (int, optional): Random seed for LASSO model. Defaults to RANDOM_STATE.

    Returns:
        best_lambda: hyperparameter with the lowest risk function value.
        best_model: fitted LASSO model with best_lambda, max_iter and random_state.
        list_error (optional): list of the value of the risk function for each tested value.
    """

    risk_values = []
    list_models = []
    if return_errors:
        list_error  = []

    for lmbda in tqdm(lambda_values, disable=no_progress):
        model = Lasso(alpha=lmbda, max_iter=max_iter, random_state = random_state)
        model.fit(X, y)
        estimated_risk = risk_function(model, X, y, weight = weight)
        risk_values.append(estimated_risk)
        list_models.append(model)
        if return_errors:
            error = mean_squared_error(y, model.predict(X))
            list_error.append(error)
        if verbose:
            print(f"Lambda value: \t{lmbda}", f"Risk function: \t{estimated_risk}", sep = "\n")
    
    best_lambda = lambda_values[np.argmin(risk_values)]
    best_model  = list_models[np.argmin(risk_values)]

    if return_errors:
        return best_lambda, best_model, list_error
    else:
        return best_lambda, best_model


def epoch_LASSO(X, y, lambda_values, weight="aic", max_iter = MAX_ITER, random_state = RANDOM_STATE, no_progress = True):
    """Perfoms a run of the best parameter finding with LASSO model.

    Args:
        X: set of features.
        y: set of dependent variables.
        lambda_values (list): list of penalization parameters to test.
        weight (str, optional): weight for the function, either 'aic' or 'bic'. Defaults to 'aic'.
        max_iter (int, optional): maximum number of iterations of the LASSO function. Defaults to MAX_ITER.
        random_state (int, optional): Random seed for LASSO model. Defaults to RANDOM_STATE.

    Returns:
        lmbda: optimal penalization parameter.
        selected_predictors: list of the non-zero parameters index.
        coefficients: list of all the parameters' coefficients. 
    """

    lmbda, model = finding_best_parameter_LASSO(
        X = X, 
        y = y, 
        weight=weight, 
        lambda_values = lambda_values, 
        return_errors = False, 
        verbose = False, 
        no_progress=no_progress, 
        max_iter = max_iter, 
        random_state = random_state
        )
    
    coefficients = model.coef_      # value to the coefficients
    selected_predictors = np.where(coefficients!= 0)[0] # index of optimal coefs

    return lmbda, selected_predictors, coefficients


def bootstrap_LASSO_RISK(X, y, lambda_values, weight="aic", n_bootstraps=100, max_iter = MAX_ITER, random_state = RANDOM_STATE, no_progress = False):
    """
    Perform bootstrap enhanced LASSO with optimal hyperparameter finding using risk function.

    Args:
        X: set of features.
        y: set of dependent variables.
        lambda_values (list): list of penalization parameters to test.
        weight (str, optional): weight for the function, either 'aic' or 'bic'. Defaults to 'aic'.
        n_bootstraps (int, optional): number of simulated samples. Defaults to 100.
        verbose (bool, optional): indications during training. Defaults to False.
        max_iter (int, optional): maximum number of iterations of the LASSO function. Defaults to MAX_ITER.
        random_state (int, optional): Random seed for LASSO model. Defaults to RANDOM_STATE.
    
    Returns:
        bootstrap_results (list): frequency of selection of each feature.
        coef_boot (list): average value of coefficients of each feature.
    """

    n_samples = X.shape[0]
    n_features = X.shape[1]
    bootstrap_results = np.zeros(n_features)
    coef_boot = [0 for _ in range(X.shape[1])]
    
    for _ in tqdm(range(n_bootstraps), disable=no_progress):

        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]

        _, selected_predictors, coefficients = epoch_LASSO(
            X = X_bootstrap, 
            y = y_bootstrap, 
            weight=weight, 
            lambda_values = lambda_values, 
            max_iter = max_iter, 
            random_state = random_state,
            no_progress= True,
            )

        coef_boot = list(map(lambda x, y: x + y, coef_boot, coefficients)) # sum all the coefficients at each step, to take average

        # Update the frequency of each variable being selected
        bootstrap_results[selected_predictors] += 1
    
    # Normalize the frequencies
    bootstrap_results /= n_bootstraps
    coef_boot = list(map(lambda x: x/len(coef_boot), coef_boot))
    
    return bootstrap_results, coef_boot


def finding_best_parameter_ELASTIC(X, y, lambda_values, l1_ratio = L1_RATIO, weight = "aic", return_errors = False, verbose = False, no_progress = True, max_iter = MAX_ITER, random_state = RANDOM_STATE):
    """Given a set of parameters, compute the value of the risk function for each of them and report the optimal penalization parameter.

    Args:
        X: set of features.
        y: set of dependent variables.
        lambda_values (list): list of penalization parameters to test.
        l1_ratio (list): list of the ratio between the two penalization parameters.
        weight (str, optional): weight for the function, either 'aic' or 'bic'. Defaults to 'aic'.
        return_errors (bool, optional): if True, returns the list of the errors for each tested parameter. Defaults to False.
        verbose (bool, optional): if True, adds some information during training. Defaults to False.
        no_progress (bool, optional): if True, does not display the progress bar. Defaults to True.
        max_iter (int, optional): maximum number of iterations of the LASSO function. Defaults to MAX_ITER.
        random_state (int, optional): Random seed for LASSO model. Defaults to RANDOM_STATE.

    Returns:
        best_lambda: hyperparameter with the lowest risk function value.
        best_model: fitted LASSO model with best_lambda, max_iter and random_state.
        list_error (optional): list of the value of the risk function for each tested value.
    """

    risk_values = []
    list_models = []
    if return_errors:
        list_error  = []

    parameters = [(lam, mu) for lam, mu in product(lambda_values, l1_ratio)]
    
    for (lmbda, mu) in tqdm(parameters, disable=no_progress):
        for mu in l1_ratio:
            model = ElasticNet(
                alpha = lmbda,
                l1_ratio = mu,
                max_iter = max_iter,
                random_state = random_state,
            )
            
        model.fit(X, y)
        estimated_risk = risk_function(model, X, y, weight = weight)
        risk_values.append(estimated_risk)
        list_models.append(model)
        if return_errors:
            error = mean_squared_error(y, model.predict(X))
            list_error.append(error)
        if verbose:
            print(f"Lambda value: \t{lmbda}", f"Risk function: \t{estimated_risk}", sep = "\n")
    
    best_lambda = parameters[np.argmin(risk_values)][0]
    best_model  = list_models[np.argmin(risk_values)]

    if return_errors:
        return best_lambda, best_model, list_error
    else:
        return best_lambda, best_model


def epoch_ELASTIC(X, y, lambda_values, l1_ratio = L1_RATIO, weight="aic", max_iter = MAX_ITER, random_state = RANDOM_STATE, no_progress = True):
    """Perfoms a run of the best parameter finding with LASSO model.

    Args:
        X: set of features.
        y: set of dependent variables.
        lambda_values (list): list of penalization parameters to test.
        weight (str, optional): weight for the function, either 'aic' or 'bic'. Defaults to 'aic'.
        max_iter (int, optional): maximum number of iterations of the LASSO function. Defaults to MAX_ITER.
        random_state (int, optional): Random seed for LASSO model. Defaults to RANDOM_STATE.

    Returns:
        lmbda: optimal penalization parameter.
        selected_predictors: list of the non-zero parameters index.
        coefficients: list of all the parameters' coefficients. 
    """

    lmbda, model = finding_best_parameter_ELASTIC(X = X, y = y, weight=weight, lambda_values = lambda_values, l1_ratio = l1_ratio, return_errors = False, verbose = False, no_progress=no_progress, max_iter = max_iter, random_state = random_state)

    coefficients = model.coef_           # value to the coefficients
    selected_predictors = np.where(coefficients!= 0)[0] # index of optimal coefs

    return lmbda, selected_predictors, coefficients


def bootstrap_ELASTIC_RISK(X, y, lambda_values, l1_ratio = L1_RATIO, weight="aic", n_bootstraps=100, max_iter = MAX_ITER, random_state = RANDOM_STATE, no_progress = False):
    """
    Perform bootstrap enhanced LASSO with optimal hyperparameter finding using risk function.

    Args:
        X: set of features.
        y: set of dependent variables.
        lambda_values (list): list of penalization parameters to test.
        weight (str, optional): weight for the function, either 'aic' or 'bic'. Defaults to 'aic'.
        n_bootstraps (int, optional): number of simulated samples. Defaults to 100.
        verbose (bool, optional): indications during training. Defaults to False.
        max_iter (int, optional): maximum number of iterations of the LASSO function. Defaults to MAX_ITER.
        random_state (int, optional): Random seed for LASSO model. Defaults to RANDOM_STATE.
    
    Returns:
        bootstrap_results (list): frequency of selection of each feature.
        coef_boot (list): average value of coefficients of each feature.
    """

    n_samples = X.shape[0]
    n_features = X.shape[1]
    bootstrap_results = np.zeros(n_features)
    coef_boot = [0 for _ in range(X.shape[1])]
    
    for _ in tqdm(range(n_bootstraps), disable = no_progress):

        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]

        _, selected_predictors, coefficients = epoch_ELASTIC(
            X = X_bootstrap, 
            y = y_bootstrap, 
            weight=weight, 
            lambda_values = lambda_values,
            l1_ratio= l1_ratio,
            max_iter = max_iter, 
            random_state = random_state
            )

        coef_boot = list(map(lambda x, y: x + y, coef_boot, coefficients)) # sum all the coefficients at each step, to take average

        # Update the frequency of each variable being selected
        bootstrap_results[selected_predictors] += 1
    
    # Normalize the frequencies
    bootstrap_results /= n_bootstraps
    coef_boot = list(map(lambda x: x/len(coef_boot), coef_boot))
    
    return bootstrap_results, coef_boot


def run_model_RISK(data_path = "../Data/final_data_198501_200001.csv", tolerance = 0.5, weight = "aic",
              cv_lasso = None, boot_lasso = None, lambda_values = None, nb_boot_lasso = 100, 
              cv_elastic = None, l1_ratio=L1_RATIO, boot_elastic = None, nb_boot_elastic = 100, 
              verbose = None, random_state = RANDOM_STATE, max_iter = MAX_ITER):
    
    """Function that runs the models. Can be modulated so it only runs specific combinations of models (LASSO/Enet, with or w/o bootstrapping)"""
    
    X, y, predictor_name = prepare_data(order_columns(pd.read_csv(data_path)))

    X_std, y_std, _ = scale_all_data(X = X, y=y.ravel())

    date = data_path[3:].split(".")[0].split("/")[1].split("_")
    year_start, year_end = date[-2], date[-1]

    if cv_lasso:
        print(f"\n------------------------ CROSS-VALIDATION LASSO - PERIOD {year_start[0:4]} - {year_end[0:4]} ------------------------")
        lambda_lasso_risk_cv, selected_predictors_lasso_risk_cv, coefficients_lasso_risk_cv = epoch_LASSO(
            X = X_std, 
            y = y_std, 
            lambda_values = lambda_values, 
            weight = weight, 
            max_iter = max_iter, 
            random_state = random_state, 
            no_progress = False
            )
        

        dic_results_lasso_risk_cv = bootstrap_result_to_dic(coefficients_lasso_risk_cv, predictor_name)
        if verbose:
            disp_bootstrap_weights(dic_results_lasso_risk_cv, predictor_name)

        final_pred_lasso_risk_cv = predictor_selection(dic_results_lasso_risk_cv, tol = tolerance)
        
        save_list_preds_csv(final_pred_lasso_risk_cv, f"output/list_predictors_lasso_risk_cv_{year_start}_{year_end}.csv")
        print(f"Selection of predictors ({len(final_pred_lasso_risk_cv)} kept from {len(dic_results_lasso_risk_cv)})\n", final_pred_lasso_risk_cv)

    if boot_lasso:
        print(f"\n-------------------------- BOOTSTRAPPED LASSO - PERIOD {year_start[0:4]} - {year_end[0:4]} --------------------------")
        results_lasso_risk_boot, coeffs_lasso_risk_boot = bootstrap_LASSO_RISK(
            X = X_std, 
            y = y_std,
            lambda_values = lambda_values,
            weight = weight, 
            n_bootstraps = nb_boot_lasso, 
            max_iter = max_iter, 
            random_state = random_state, 
            no_progress = False
            )
        
        dic_results_lasso_risk_boot = bootstrap_result_to_dic(results_lasso_risk_boot, predictor_name)

        if verbose:
            disp_bootstrap_weights(dic_results_lasso_risk_boot, predictor_name)

        final_pred_lasso_risk_boot = predictor_selection(dic_results_lasso_risk_boot, tol=tolerance)
        save_list_preds_csv(final_pred_lasso_risk_boot, f"output/list_predictors_lasso_bootstrap_risk_{year_start}_{year_end}.csv")
        print(f"Selection of predictors ({len(final_pred_lasso_risk_boot)} kept from {len(coeffs_lasso_risk_boot)})\n", final_pred_lasso_risk_boot)

    if cv_elastic:
        print(f"\n------------------------ CROSS-VALIDATION ELASTIC - PERIOD {year_start[0:4]} - {year_end[0:4]} ------------------------")
        lambda_elastic_risk_cv, selected_predictors_elastic_risk_cv, coefficients_elastic_risk_cv = epoch_ELASTIC(
            X = X_std, 
            y = y_std,
            lambda_values = lambda_values, 
            l1_ratio = l1_ratio,
            weight = weight, 
            max_iter = max_iter, 
            random_state = random_state, 
            no_progress = False
            )
        
        dic_results_elastic_risk_cv = bootstrap_result_to_dic(coefficients_elastic_risk_cv, predictor_name)
        if verbose:
            disp_bootstrap_weights(dic_results_elastic_risk_cv, predictor_name)

        final_pred_elastic_risk_cv = predictor_selection(dic_results_elastic_risk_cv, tol = tolerance)
        
        save_list_preds_csv(final_pred_elastic_risk_cv, f"output/list_predictors_elastic_risk_cv_{year_start}_{year_end}.csv")
        print(f"Selection of predictors ({len(final_pred_elastic_risk_cv)} kept from {len(dic_results_elastic_risk_cv)})\n", final_pred_elastic_risk_cv)

    if boot_elastic:
        print(f"\n-------------------------- BOOTSTRAPPED ELASTIC - PERIOD {year_start[0:4]} - {year_end[0:4]} --------------------------")
        results_elastic_risk_boot, coeffs_elastic_risk_boot = bootstrap_ELASTIC_RISK(
            X = X_std, 
            y = y_std,
            lambda_values = lambda_values,
            l1_ratio = l1_ratio,
            weight = weight, 
            n_bootstraps = nb_boot_elastic, 
            max_iter = max_iter, 
            random_state = random_state, 
            no_progress = False
            )

        
        dic_results_elastic_risk_boot = bootstrap_result_to_dic(results_elastic_risk_boot, predictor_name)

        if verbose:
            disp_bootstrap_weights(dic_results_elastic_risk_boot, predictor_name)

        final_pred_elastic_risk_boot = predictor_selection(dic_results_elastic_risk_boot, tol=tolerance)
        save_list_preds_csv(final_pred_elastic_risk_boot, f"output/list_predictors_elastic_bootstrap_risk_{year_start}_{year_end}.csv")
        print(f"Selection of predictors ({len(final_pred_elastic_risk_boot)} kept from {len(coeffs_elastic_risk_boot)})\n", final_pred_elastic_risk_boot)

def full_run(test = False):
    
    create_folders()

    # list_data = ["Data/final_data_198501_200001.csv", "Data/final_data_199501_201001.csv", "Data/final_data_200501_202001.csv"]
    list_data = ["Data/final_data_199501_201001.csv"]
    for file_path in list_data:
        if test:
            run_model_RISK(
                data_path = file_path, 
                tolerance = 0, 
                weight = "aic",
                cv_lasso = True, 
                boot_lasso = True, 
                lambda_values = [0.001, 0.02], 
                nb_boot_lasso = 5, 
                cv_elastic = True, 
                l1_ratio = [0.05, 0.5], 
                boot_elastic = True,
                nb_boot_elastic = 2, 
                verbose = True, 
                random_state = RANDOM_STATE, 
                max_iter = MAX_ITER
                )
        else:
            run_model_RISK(
                data_path = file_path, 
                tolerance = 0, 
                weight = "aic",
                cv_lasso = False,
                boot_lasso = 50,
                lambda_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                nb_boot_lasso = 50,
                cv_elastic = False,
                l1_ratio = L1_RATIO, 
                boot_elastic = True,
                nb_boot_elastic = 50,
                verbose = True, 
                random_state = RANDOM_STATE, 
                max_iter = MAX_ITER
                )
        
    plot_selected_features()


def plot_selected_features():
    
    name_lists = []
    name_periods = []

    all_selected_features = []

    for method in METHODS:
        for date in DATES:
            file_name = f'output/list_predictors_{method}_bootstrap_risk_{date}.csv'

            try:
                file = pd.read_csv(file_name)
                selected_features = file.columns.tolist()
            except:
                selected_features = []

            exec(f'{method}_{date} = selected_features')
            x = f"{method}_{date}"
            name_periods.append(x)
            exec(f'name_lists.append({method}_{date})')
            new = [pred for pred in selected_features if pred not in all_selected_features]
            all_selected_features += new


    data_heatmap = {}
    data_heatmap['predictors'] = all_selected_features

    for idx, period in enumerate(name_lists):
        data_heatmap[name_periods[idx]] = [1 if pred in period else 0 for pred in all_selected_features]

    dataframe_heatmap = pd.DataFrame(
        data = data_heatmap
    )

    dataframe_heatmap.set_index(
        'predictors', 
        inplace = True, 
        drop = True
    )

    plt.figure(figsize=(15,5))
    sns.heatmap(
        data = dataframe_heatmap.T,
        cbar=False,

    )
    plt.xlabel('')
    plt.savefig(f"output/figures/partA2_selected_features")
    plt.show()



if __name__ == "__main__":

    full_run(test = True)