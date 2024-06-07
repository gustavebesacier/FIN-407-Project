import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from sklearn.linear_model import Lasso
import pandas as pd

def group_lasso_penality(lbd, beta, rows, columns):
    """
    The first group penality construction
    :param lbd: penality term
    :param beta: estimator
    :param rows: nbr of rows
    :param columns: nbr of columns
    :return: the penality evaluation
    """
    beta_reshaped = beta.reshape((rows, columns), order='F')
    beta_squared = np.power(beta_reshaped, 2)
    column_sums = np.sum(beta_squared, axis=0)
    penalty = lbd * np.sum(np.sqrt(column_sums))
    return penalty

def expected_return(X_extended, beta_flat):
    """
    Computes the expected return
    :param X_extended: The matrix of observation with splines extension
    :param beta_flat: estimator
    :return: expected return
    """
    exp_ret = np.dot(X_extended, beta_flat).flatten()
    return exp_ret

def objective_function(beta, X_extended, ret, lbd, rows, columns):
    """
    The objective function of the first group Lasso
    :param beta: estimator
    :param X_extended: The matrix of observation with splines extension
    :param ret: true return (y target)
    :param lbd: penalty
    :param rows: number of rows
    :param columns: number of columns
    :return: value of the objective function
    """
    err = ret - expected_return(X_extended,beta)
    res = np.sum(np.power(err, 2))
    penalty = group_lasso_penality(lbd, beta, rows, columns)
    return res + penalty

def weight_gen(pre_betas, rows, columns, threshold):
    """
    Generates the weights for the second group Lasso
    :param pre_betas: estimator fond by the first group Lasso
    :param rows: nbr of rows
    :param columns: nbr of columns
    :param threshold: threshold to assess the value is too small
    :return:
    """
    large_value = 1e10
    pre_beta_reshaped = pre_betas.reshape((rows, columns), order='F')
    column_sums = np.sum(np.power(pre_beta_reshaped, 2), axis=0)
    # Create a new vector w with the same shape as column_sums, initialized to np.inf
    predictor_weight = np.full(columns,large_value)
    # Replace values in weights based on the condition
    non_zero_indices = np.abs(column_sums) > threshold
    predictor_weight[non_zero_indices] = np.power(column_sums[non_zero_indices], -0.5)
    return predictor_weight

def sec_group_lasso_penality(weights, lbd, beta, rows, columns):
    """
    The second group penality construction (utilized the weights)
    :param lbd: penality term
    :param beta: estimator
    :param rows: nbr of rows
    :param columns: nbr of columns
    :param weights: The weights to discard even more predictors
    :return: the penality evaluation
    """
    beta_reshaped = beta.reshape((rows, columns), order='F')
    beta_squared = np.power(beta_reshaped, 2)
    column_sums = np.sum(beta_squared, axis=0)
    penalizer = np.sum(np.sqrt(column_sums * weights))
    return penalizer*lbd

def sec_objective_function(beta, X_extended, ret, weights, lbd, rows, columns):
    """
    The objective function of the second group Lasso
    :param beta: estimator
    :param X_extended: The matrix of observation with splines extension
    :param ret: true return (y target)
    :param weights: the weights to scalke the predictors
    :param lbd: penalty
    :param rows: number of rows
    :param columns: number of columns
    :return: value of the objective function
    """
    err = ret - expected_return(X_extended,beta)
    res = np.sum(np.power(err, 2))
    penalty = sec_group_lasso_penality(weights, lbd, beta, rows, columns)
    return res + penalty

# Define the spline basis functions
def spline_basis(c, knots, L):
    """
    The spline function to extent the model towards non-parametric
    :param c: value of the predictor
    :param knots: nbr of knots the function has to go through
    :param L: nbr of extension + 2
    :return: modified predictor
    """
    basis = np.zeros((len(c), L + 2))
    basis[:, 0] = 1
    basis[:, 1] = c
    basis[:, 2] = c**2
    for k in range(3, L + 2):
        basis[:, k] = np.maximum(c - knots[k-3], 0)**2
    return basis

def extend(data, knots, L):
    """
    Extends my dataframe with the spline formulation
    :param data: dataframe
    :param knots: nbr of knots the function has to go through
    :param L: nbr of extension +2
    :return: modified dataframe
    """
    cols = data.columns.drop(["yyyymm","permno"])
    data_ranked_trans = data[["permno","yyyymm"]].copy()
    for column in cols:
        data_spline = pd.DataFrame(spline_basis(data[column], knots, L), columns=[f'{column}_{k+1}' for k in range(L + 2)])
        data_ranked_trans = pd.concat([data_ranked_trans, data_spline], axis=1)
    return data_ranked_trans

def NPE(X,y, lbd1, lbd2, L=1, threshold = 1e-15):
    """
    This function runs the adaptive group Lasso non-parametric regression
    :param X: Data observed
    :param y: Target
    :param lbd1: regularization term for the first Lasso
    :param lbd2: regularization term for the second Lasso
    :param L: nbr of extension + 2
    :param threshold: nbr at which a estimator is considered 0
    :return: list of betas, list of r2 and mse for each date
    """
    dates = X["yyyymm"].unique()
    selected_indices_all = []
    mse_list = []
    r2_list = []
    beta_list = []
    indicices_per_beta = []
    beta_rows = L + 2  # differnet number of stocks for that time
    beta_columns = len(X.columns.drop(["yyyymm","permno"]))  # different number of predictors

    # Define knots and interval
    knots = np.linspace(0, 1, L + 1)

    #Build extension
    X_extended = extend(X, knots, L)

    for date in tqdm(dates):
        # Subset data for the current date
        subset = X_extended[X_extended["yyyymm"] == date]
        y_subset = y[y["yyyymm"] == date]

        X_raw = subset.drop(columns=["yyyymm", "permno"]).to_numpy()
        y_raw = y_subset.drop(columns=["yyyymm"]).to_numpy().ravel()


        #beta_matrix = np.random.rand(beta_rows, beta_columns)
        beta_matrix = 2 * np.random.rand(beta_rows, beta_columns) - 1
        #beta_matrix = np.zeros(shape=(beta_rows, beta_columns))
        #beta_matrix = np.full(shape=(beta_rows, beta_columns), fill_value=2)
        beta_0 = beta_matrix.flatten(order='F')
        # Minimize the objective function
        #result = minimize(objective_function, beta_0, args=(X_raw,y_raw,lbd1, beta_rows, beta_columns), method= "Nelder-Mead")
        result = minimize(objective_function, beta_0, args=(X_raw, y_raw, lbd1, beta_rows, beta_columns))

        # Extract the optimized beta
        beta_1 = result.x
        #print(beta_1)

        beta1_reshaped = beta_1.reshape((beta_rows, beta_columns), order='F')
        column_sums = np.sum(np.power(beta1_reshaped, 2), axis=0)

        zero_indices = np.abs(column_sums) < threshold
        indicices_per_beta.append(zero_indices)
        weights = np.power(column_sums[~zero_indices], -0.5)

        mask = np.tile(zero_indices, (L + 2, 1)).flatten(order='F')
        X_filtered = X_raw[:, ~mask]

        #print(len(weights))
        #print(f"weights are {weights}")

        beta_columns_filtered = len(weights)

        #Minimize the objective function
        beta_matrix = np.zeros(shape = (beta_rows, beta_columns_filtered))
        beta_01 = beta_matrix.flatten(order='F')
        #sec_result = minimize(sec_objective_function, beta_01, args=(X_filtered, y_raw ,weights, lbd2, beta_rows, beta_columns_filtered), method= "Nelder-Mead")
        sec_result = minimize(sec_objective_function, beta_01,
                              args=(X_filtered, y_raw, weights, lbd2, beta_rows, beta_columns_filtered))

        beta_2 = sec_result.x
        #print(beta_2)

        zero_indices_final = np.abs(beta_2) < threshold

        # Set those elements to zero
        beta_2[zero_indices_final] = 0

        pred = expected_return(X_filtered,beta_2)

        #Calculate the variance
        error = pred-y_raw
        mean_error = np.sum(error)

        #Calculate performance metrics
        mse = mean_squared_error(y_raw, pred)
        r2 = r2_score(y_raw, pred)

        mse_list.append(mse)
        r2_list.append(r2)
        beta_list.append(beta_2)
        print(
            f"before cutoff the first lasso we had {beta_columns} predictors, it then dropped to {len(weights)} and the second lasso gave {len(np.where(beta_2 != 0)[0]/(L + 2))} predictors")
        print(f"mean squared error is {mse}")
        print(f"the r squared is {r2}")


    return beta_list, mse_list, r2_list


#########################################################
#########################################################
#########################################################
#The following are old implementation
#########################################################
#########################################################
#########################################################
def NonParamEst(X,y, lbd1, lbd2, L=1, threshold = 1e-10):
    """
    OLD DO NOT USE
    This is an old function where I tried modifying the lasso class
    Please disregarde
    :param X:
    :param y:
    :param lbd1:
    :param lbd2:
    :param L:
    :param threshold:
    :return:
    """
    dates = X["yyyymm"].unique()
    selected_indices_all = []
    mse_list = []
    r2_list = []
    beta_list = []
    indicices_per_beta = []


    for date in tqdm(dates):
        # Subset data for the current date
        subset = X[X["yyyymm"] == date]
        y_subset = y[y["yyyymm"] == date]

        X_raw = subset.drop(columns=["yyyymm", "permno"]).to_numpy()
        y_raw = y_subset.drop(columns=["yyyymm"]).to_numpy().ravel()

        shape = X_raw.shape
        rows = 3 #differnet number of stocks for that time
        columns = shape[1] #different number of predictors

        X_extended = apply_basis(X_raw)

        #Run the first Lasso group
        beta_matrix = np.random.rand(rows, columns)
        beta_0 = beta_matrix.flatten(order='F')

        # Run the first Lasso regression
        lasso1 = Lasso(alpha=lbd1, fit_intercept=False, max_iter=10000)
        lasso1.fit(X_extended, y_raw)
        beta_1 = lasso1.coef_

        beta1_reshaped = beta_1.reshape((rows, columns), order='F')
        column_sums = np.sum(np.power(beta1_reshaped, 2), axis=0)

        zero_indices = np.abs(column_sums) < threshold
        indicices_per_beta.append(zero_indices)

        # Drop columns from X_raw where zero_indices is True
        X_filtered = X_raw[:, ~zero_indices]
        columns = X_filtered.shape[1]  # different number of predictors

        weights = np.power(column_sums[~zero_indices], -0.5)

        X_filtred_extended = apply_basis(X_filtered)

        #[non_zero_indices] = np.power(column_sums[non_zero_indices], -0.5)

        #print(beta_1)

        #pred = expected_return(X_extended, beta_1)

        #mse = mean_squared_error(y_raw, pred)
        #r2 = r2_score(y_raw, pred)

        # Minimize the objective function
        #result = minimize(objective_function, beta_0, args=(X_extended,y_raw,lbd1, rows, columns))

        # Extract the optimized beta
        #optimized_beta = result.x

        #weight building
        #weights = weight_gen(beta_1, rows, columns, threshold)

        #Generate weights based on the first Lasso coefficients
        #weights = weight_gen(beta_1, rows, columns, threshold)



        # Minimize the objective function
        #sec_result = minimize(sec_objective_function, beta_0, args=(X_extended, y_raw ,weights, lbd2, rows, columns))

        #best_est = sec_result.x
        # Custom penalty term for the second Lasso regression
        class WeightedLasso(Lasso):
            def __init__(self, alpha=1.0, fit_intercept=True, max_iter=10000, tol=1e-5):
                super().__init__(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)

            def fit(self, X, y):
                super().fit(X, y)
                coef_reshaped = self.coef_.reshape((rows, columns), order='F')
                penalty = lbd2 * np.sum(
                    [weights[k] * np.sqrt(np.sum(coef_reshaped[:, k] ** 2)) for k in range(columns)])
                self.coef_ -= penalty
                return self

        # Run the second Lasso regression
        lasso2 = WeightedLasso(alpha=lbd2, fit_intercept=False)
        lasso2.fit(X_filtred_extended, y_raw)
        beta_2 = lasso2.coef_

        selected_indices_all.append(np.where(beta_2 != 0)[0])
        mse_list.append(np.mean((lasso2.predict(X_filtred_extended) - y_raw) ** 2))
        r2_list.append(lasso2.score(X_filtred_extended, y_raw))
        beta_list.append(beta_2)

        #zero_indices = np.abs(beta_2) < threshold

        print(f"before cutoff the first lasso we had {X_raw.shape[1]} predictors, it then dropped to {len(weights)} and the second lasso gave {len(np.where(beta_2 != 0)[0])} coefficient from {3*len(weights)}")
        print(f"for {date}, the coefficient are the followings:{beta_2}")

        #selected_indices_all.append(best_est)

        #pred = expected_return(X_extended,best_est)

        #Calculate the variance
        #error = pred-y_raw
        #mean_error = np.sum(error)


        # Calculate performance metrics
        #mse = mean_squared_error(y_raw, pred)
        #r2 = r2_score(y_raw, pred)

        #mse_list.append(mse)
        #r2_list.append(r2)
    mse_min = np.inf
    r2_min = np.inf
    obs = len(dates)
    beta_best_mse = []
    beta_best_mse = []

    for i in range(obs):
        beta = beta_list[i]
        null_indices = indicices_per_beta[i]
        mse = 0
        r2 = 0
        for date in tqdm(dates):
            # Subset data for the current date
            subset = X[X["yyyymm"] == date]
            y_subset = y[y["yyyymm"] == date]

            X_raw = subset.drop(columns=["yyyymm", "permno"]).to_numpy()
            y_raw = y_subset.drop(columns=["yyyymm"]).to_numpy().ravel()

            X_filtered = X_raw[:, ~null_indices]
            X_filtred_extended = apply_basis(X_filtered)
            pred = expected_return(X_filtred_extended, beta)
            mse += mean_squared_error(y_raw, pred)
            r2 += r2_score(y_raw, pred)

        mse /= obs
        r2 /= obs

        if r2< r2_min:
            beta_best_r2 = beta
            r2_min = r2

        if mse < mse_min:
            beta_best_mse = beta
            mse_min = mse



    return selected_indices_all, mse_list, r2_list, beta_best_r2, beta_best_mse, r2_min, mse_min


#Rank tranformation
def rank_transform(df):
    """
    OLD DO NOT USE
    Applies the rank transformation per columns
    :param df: dataframe
    :return: ranked transformed dataframe
    """
    columns = df.columns.drop(["yyyymm", "permno"])
    df_ranked_trans = df.copy()
    df_ranked_trans['N_t'] = df_ranked_trans.groupby('yyyymm')['permno'].transform('count')

    for column in columns:
        df_ranked_trans[column] = df.groupby('yyyymm')[column].rank(method='first')

        # Normalize the rank by the number of stocks per period
        df_ranked_trans[column] = df_ranked_trans[column] / (df_ranked_trans['N_t'] + 1)
        # Drop the intermediate column

    df_ranked_trans = df_ranked_trans.drop(columns=['N_t'])

    return df_ranked_trans

#Group Lasso Regularized Estimator

#Build the splines functions
def basis_functions(c, knots):
    """
    OLD DO NOT USE
    Generate the basis functions for quadratic splines.
    :param c: Input variable (single value)
    :param knots: List of knot points
    :return: List of evaluated basis functions at c
    """
    p = [1, c, c ** 2]  # p1(c) = 1, p2(c) = c, p3(c) = c^2
    p += [max(0, (c - t) ** 2) for t in knots]  # pk(c) = max{c - t, 0}^2 for k > 3
    return np.array(p)

# let us build the first method to find the coefficients
def apply_basis(X_raw):
    """
    Old function to apply the simple splines
    :param X_raw:
    :return:
    """
    shape = X_raw.shape
    ones = np.ones(shape)
    squared = np.power(X_raw, 2)

    # Initialize an empty array to store the extended feature matrix
    X_extended = np.empty((shape[0], shape[1] * 3))

    # Alternate columns of ones, self, and squared
    for i in range(shape[1]):
        X_extended[:, i * 3] = ones[:, i]
        X_extended[:, i * 3 + 1] = X_raw[:, i]
        X_extended[:, i * 3 + 2] = squared[:, i]

    return X_extended
