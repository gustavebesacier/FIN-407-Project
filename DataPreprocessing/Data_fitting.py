import pandas as pd
import numpy as np
from scipy.linalg import pinvh, inv
from tqdm import tqdm
from Utils.Utilities import categorical_finder

"""
This file handles replacing nans efficiently inspired from the paper Missing Financial Data written by Bryzgalova and al..
It tries to rely as much as possible on the numpy library
It builds an estimate based Cross-correctional data and with backward time series
"""

def build_empir_cov(nbr_predictors, P_matrix):
    """
    Build the empirical covariance matrix of the predictors
    :param nbr_predictors: number of predictors and size of the cov matrix
    :param P_matrix: the whole observation of predictors at some time t
    :return: empirical covariance matrix of the predictors size (nbr_predictors x nbr_predictors)
    """
    # Initialize the covariance matrix
    Cov_matrix = np.zeros((nbr_predictors, nbr_predictors))

    # Calculate the covariance matrix
    for p in range(nbr_predictors):
        for l in range(p, nbr_predictors):
            # Create boolean masks for non-NaN values
            mask = ~np.isnan(P_matrix[:, p]) & ~np.isnan(P_matrix[:, l])
            valid_values_p = P_matrix[mask, p]
            valid_values_l = P_matrix[mask, l]

            # resolve Issue when division by zero happens
            if len(valid_values_p) > 0:
                Cov_matrix[p, l] = np.dot(valid_values_p, valid_values_l) / len(valid_values_p)
                if np.isnan(Cov_matrix[p, l]):
                    print(f"NaN detected at Cov_optimized[{p}, {l}]")
                    print(f"valid_values_p: {valid_values_p}")
                    print(f"valid_values_l: {valid_values_l}")
                if p != l:
                    Cov_matrix[l, p] = Cov_matrix[p, l]

            #Check for NaNs
            if np.isnan(Cov_matrix[p, l]):
                print(f"NaN detected after assignment at Cov_optimized[{p}, {l}]")

    # Check for NaNs in the entire matrix and print their positions
    # Avoids nan appearances (weird pass)
    nan_positions = np.argwhere(np.isnan(Cov_matrix))
    if nan_positions.size > 0:
        print(f"NaN values found at positions: {nan_positions}")
    finite_check = np.isfinite(Cov_matrix)
    if not finite_check.all():
        print("Non-finite values found in the covariance matrix.")

    assert not np.isnan(Cov_matrix).any(), "There are NaN values in the cov matrix."

    return Cov_matrix

def build_loadings(empirical_cov, K_factors):
    """
    This build the weights matrix: LAMBDA
    :param empirical_cov: the empirical covariance matrix of the predictors
    :param K_factors: nbr of explenatory factors
    :return: LAMBDA
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(empirical_cov)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the K largest eigenvalues and their associated eigenvectors
    D = np.diag(sorted_eigenvalues[:K_factors])
    V = sorted_eigenvectors[:, :K_factors]
    loadings = V @ np.sqrt(D)

    # Checks for nans
    assert not np.isnan(loadings).any(), "There are NaN values in the cov matrix."

    return loadings


def build_factors(P_matrix, L_matrix, gamma, K_factors, nbr_firms, nbr_predictors):
    """

    :param P_matrix:
    :param L_matrix:
    :param gamma:
    :param K_factors:
    :param nbr_firms:
    :param nbr_predictors:
    :return:
    """
    # Create a boolean mask where non-NaN values are True and NaN values are False
    mask = ~np.isnan(P_matrix)

    # Replace True values with 1 and False values with 0
    W = mask.astype(int)

    F = np.zeros(shape=(nbr_firms, K_factors))
    regularize = gamma * np.identity(K_factors)

    # Apply rules
    for i in range(nbr_firms):
        X = np.zeros(shape=(K_factors, K_factors))
        Y = np.zeros(K_factors)
        for k in range(nbr_predictors):
            if W[i, k] != 0:
                factors_k = L_matrix[k, :]
                X += W[i, k] / nbr_predictors * np.outer(factors_k, factors_k) + regularize
                Y += W[i, k] / nbr_predictors * P_matrix[i, k] * factors_k
        if not np.all(X == 0):
            F[i, :] += inv(X) @ Y

    return F, W

def build_time_series_extension(N_t, nbr_predictors, error, P_minus_zeroed, P_hat, W_tilde, W, P):
    # Stack the matrices along a new axis to create a 3D matrix
    X_stacked = np.stack((P_hat, P_minus_zeroed, error), axis=-1)
    P_bar = np.zeros(shape=(N_t, nbr_predictors))
    for j in range(nbr_predictors):
        X = np.zeros(shape=(3, 3))
        Y = np.zeros(3)
        for n in range(N_t):
            if W[n, j] != 0 and W_tilde[n, j] != 0:
                X_vector = X_stacked[n, j, :]
                #if X_vector[-1] == 0 and X_vector[-2] == 0:
                    #X[0, 0] += X_vector[0] ** 2
                    #Y[0] += P[n, j] * X_vector[0]
                #else:
                X += np.outer(X_vector, X_vector)
                Y += P[n, j] * X_vector
        if np.isclose(X[1, 1], 0) and np.isclose(X[2, 2], 0):
            if Y[0] == 0:
                beta_j = np.zeros(3)
            else:
                beta_j = Y / X[0, 0]
            X_j = X_stacked[:, j, :]
            P_j = X_j @ beta_j
            assert P_j.shape == (N_t,)
            P_bar[:, j] = P_j
        else:
            #Avoid issue when inverting matrix
            beta_j = pinvh(X) @ Y
            X_j = X_stacked[:, j, :]
            P_j = X_j @ beta_j
            assert P_j.shape == (N_t,)
            P_bar[:, j] = P_j

    return P_bar


def Fit_using_XS_B(data, K_factors, nbr_predictors, gamma):
    data = data.replace([np.inf, -np.inf], np.nan)
    dates = data['yyyymm'].unique()
    data_per_date = data.groupby("yyyymm")

    fitted_data = pd.DataFrame()

    for i, date in enumerate(tqdm(dates)):
        # Fetch previous date if it exists
        if i > 0:
            dateminus = dates[i - 1]
        else:
            dateminus = None

        # Access the current date
        data_t = data_per_date.get_group(date)
        # Sort current dataframe by permno
        data_t_sorted = data_t.sort_values(by='permno').reset_index(drop=True)

        original_index = data_t.index
        # Drop specified columns
        data_t_dropped = data_t.drop(["yyyymm", "permno"], axis=1)

        # Count non-NaN values for each column
        #non_nan_counts = data_t_dropped.notna().sum()

        # Sort columns by non-NaN counts in descending order
        #top_predictor_columns = non_nan_counts.sort_values(ascending=False).head(nbr_predictors).index

        # Select the top nbr_predictors columns
        #data_t_top_predictors = data_t_sorted[top_predictor_columns]

        all_predictors_index = pd.Index(data_t_dropped.columns)

        data_prev = data_per_date.get_group(dateminus) if dateminus is not None else None

        # Build matrix of predictors
        P = data_t_dropped.to_numpy()
        P_shape = P.shape
        assert P_shape[1] == nbr_predictors, f"The shape of the matrix of predictors is {P.shape}"
        # Number of firms observation for time t
        N_t = P_shape[0]

        # Create P_minus matrix
        if data_prev is not None:
            data_prev_sorted = data_prev.sort_values(by='permno').reset_index(drop=True)

            # Merge previous data with current index
            merged_data = pd.merge(data_t_sorted[['permno']], data_prev_sorted, on='permno', how='left')
            assert all(
                merged_data['permno'].unique() == data_t_sorted['permno'].unique()), f"All permno are not the same!"

            # Extract relevant columns for comparison
            P_minus = merged_data[all_predictors_index].to_numpy()
        else:
            # If no previous data, fill P_minus with NaN
            P_minus = np.full_like(P, np.nan)

        assert P.shape == P_minus.shape, f"Discrepencies between the predictors shape"

        # Build the empirical covariance
        emp_cov = build_empir_cov(nbr_predictors=nbr_predictors, P_matrix=P)

        assert emp_cov.shape == (
            nbr_predictors, nbr_predictors), f"The empirical covariance matrix of predictors has shape {emp_cov.shape}"

        loadings = build_loadings(empirical_cov=emp_cov, K_factors=K_factors)

        assert loadings.shape == (
            nbr_predictors, K_factors), f"The empirical covariance matrix of predictors has shape {loadings.shape}"

        factors, W = build_factors(P_matrix=P, L_matrix=loadings, gamma=gamma, K_factors=K_factors, nbr_firms=N_t,
                                   nbr_predictors=nbr_predictors)

        assert factors.shape == (
            N_t, K_factors), f"The empirical covariance matrix of predictors has shape {loadings.shape}"

        P_hat = factors @ loadings.T

        if data_prev is not None:
            # Create a boolean mask where non-NaN values are True and NaN values are False
            mask = ~np.isnan(P_minus)

            # Replace True values with 1 and False values with 0
            W_tilde = mask.astype(int)

            # Change nan to 0 to avoid casing
            error = np.nan_to_num(P_minus - P_hat, nan=0)
            P_minus_zeroed = np.nan_to_num(P_minus, nan=0)
            P_bar = build_time_series_extension(N_t = N_t, nbr_predictors=nbr_predictors, error = error,P_minus_zeroed= P_minus_zeroed,  P_hat = P_hat, W_tilde = W_tilde, W=W, P = P)

        else :
            P_bar = np.zeros(shape=(N_t, nbr_predictors))
            W_tilde = np.zeros(shape=(N_t, nbr_predictors))

        # Replace NaN values in P with 0 before performing element-wise multiplication
        P_zeroed = np.nan_to_num(P, nan=0)

        # Fills the data using the various predictors
        P_filled = W * P_zeroed + (1 - W) * (1-W_tilde)* P_hat + (1 - W) * W_tilde * P_bar

        assert np.all(W+ (1 - W) * W_tilde+(1 - W) * (1-W_tilde) == np.ones(shape=(N_t, nbr_predictors)))

        # Check for NaNs in the entire matrix and print their positions
        nan_positions = np.argwhere(np.isnan(P_filled))
        if nan_positions.size > 0:
            print(f"NaN values found at positions: {nan_positions}")
            assert True, f"Nan values in fitted data"

        data_filled_t = pd.DataFrame(P_filled, columns=all_predictors_index, index=original_index)
        data_filled_t['permno'] = data_t['permno'].values
        data_filled_t['yyyymm'] = date

        # Append to the overall fitted data
        fitted_data = pd.concat([fitted_data, data_filled_t], ignore_index=False)

    return fitted_data

#data = pd.read_csv("../Data/shorter_1000permnos.csv")
#list_cat = categorical_finder(data)
#data = data.drop(columns=list_cat)
#fitted_data = Fit_using_XS_B(data, K_factors=25, nbr_predictors=100, gamma=0.0001)
#fitted_data.to_csv("../Data/short_XS_B_1000_fitted.csv", index=False)