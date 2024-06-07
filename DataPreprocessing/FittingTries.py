import numpy as np
from scipy.linalg import lstsq
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from DataAnalysis.Data_Handler import Firm_Extractor
import json
from tqdm import tqdm

def Fitted_data_generator(data, filter=0.3, timeline_percentage=0.075, nan_threshold=0.65, near_full_threshold=1,
                          save=True, filename="data_fitted.csv"):
    unique_permnos = data['permno'].unique().tolist()

    # Initialize an empty DataFrame for Data_fitted
    data_fitted = pd.DataFrame()

    #Dictionnary that will contain permno as key and list of full columns
    permno_dict = {}

    # Analyze and fit the data stock by stock
    for permno in tqdm(unique_permnos):
        # Extract the fitted sample for the current permno
        sample_fitted = Firm_Extractor(permnos=permno, data=data)
        sample_fitted, predictors_list = Fit_missing_values(sample_fitted, filter=filter, timeline_percentage=timeline_percentage, nan_threshold=nan_threshold,
                                           near_full_threshold=near_full_threshold)

        if sample_fitted is not None:
            # Concatenate the fitted sample with Data_fitted
            data_fitted = pd.concat([data_fitted, sample_fitted], ignore_index=True)

            # Add the permno and predictors_list to the permno_dict
            permno_dict[permno] = predictors_list

    # Fill missing columns in Data_fitted with NaN values
    missing_columns = set(data.columns) - set(data_fitted.columns)
    for column in missing_columns:
        data_fitted[column] = np.nan

    # Save the file if needed
    if save:
        data_fitted.to_csv(filename, index=False)

        # Save permno_dict to a JSON file
        with open('../Data/permno_dict.json', 'w') as json_file:
            json.dump(permno_dict, json_file)

    return data_fitted

def Remove_Extremities(data, timeline_percentage = 0.075, threshold=0.65):
    #retrieve the nbr of values we will test
    timeline_length = len(data)
    nbr_predictors = len(data.columns)
    threshold_index = int(timeline_length*timeline_percentage)
    print("There are ", timeline_length,"observations of this stock")

    #Retrieve the beginning and the ending of our stock timeline
    begin_seq = data.iloc[:threshold_index]
    end_seq = data.iloc[-threshold_index:]

    max_nbr_nan = threshold * nbr_predictors

    # Check if each row within the sequences has only a percentage of non-null values below the threshold
    begin_seq_drop = begin_seq[begin_seq.isnull().sum(axis=1) >= max_nbr_nan]
    end_seq_drop = end_seq[end_seq.isnull().sum(axis=1) >= max_nbr_nan]

    #Drop the rows from the original data
    data = data.drop(begin_seq_drop.index)
    data = data.drop(end_seq_drop.index)
    print("After removal of extremities, there are ", len(data), "observations")

    return data

def Drop_Columns(data, threshold):
    stock_length = len(data)
    nbr_predictors = len(data.columns)
    print("Total number of predictors before removal is:", nbr_predictors)

    # Check for columns filled with inf values
    inf_columns = data.columns[data.isin([np.inf]).any()].tolist()

    # Iterate over inf columns
    for col in inf_columns:
        inf_ratio = data[col].isin([np.inf]).sum() / stock_length
        if inf_ratio >= threshold:
            # Drop the column if inf ratio exceeds threshold
            data.drop(col, axis=1, inplace=True)
            print(f"Dropped column '{col}' due to exceeding threshold of {threshold}% inf values.")
        else:
            # Replace inf values with NaN otherwise
            data[col].replace([np.inf], np.nan, inplace=True)
            print(f"Replaced inf values with NaN in column '{col}'.")

    # Drop columns with too many NaNs
    data.dropna(axis=1, thresh=threshold * stock_length, inplace=True)
    print("Total number of predictors after removal is:", len(data.columns))
    return data

def data_summary(data, threshold = 1):
    # Print the number of columns with no missing values and their names
    no_missing_values_cols = data.columns[data.notnull().all()]
    print("Number of columns with no missing values:", len(no_missing_values_cols))
    print("Columns with no missing values:", no_missing_values_cols.tolist())

    # Calculate the percentage of missing values for each column
    missing_percentage = data.isnull().mean() * 100

    # Filter columns where the missing percentage is 10% or less
    few_missing_values_cols = missing_percentage[missing_percentage <= threshold]

    # Print the number of columns with few missing values and their names
    print("Number of columns with",threshold,"% or less missing values:", len(few_missing_values_cols))
    print("Columns with",threshold,"% or less missing values:", few_missing_values_cols.index.tolist())

    return no_missing_values_cols.tolist(), few_missing_values_cols.index.tolist()


def Fit_missing_values(id_data, filter=0.3, timeline_percentage = 0.075, nan_threshold=0.65, near_full_threshold = 1):
    modified_data = id_data.copy()

    #Print a summary of the stock in concern
    _, _ = data_summary(data=modified_data, threshold=near_full_threshold)

    #Deletes companies with few observations
    if len(modified_data)<=6:
        return None, None

    #Delete columns depending on their number of empty values
    modified_data = Drop_Columns(data=modified_data, threshold=filter)

    #Takes care of missing valeus in the extremeties as they are generally more empty than the rest of the data
    modified_data = Remove_Extremities(data=modified_data,timeline_percentage=timeline_percentage, threshold=nan_threshold)

    # Print a summary of the stock after removal
    full_col, near_full_col = data_summary(modified_data,threshold=near_full_threshold)

    # Define the list of essential column names

    # Check if near_full_col contains only permno and yyyymm as they are useless to help us predict data
    while set(near_full_col) == {"permno", "yyyymm"} and near_full_threshold <= 30:
        near_full_threshold += 1
        full_col, near_full_col = data_summary(modified_data, threshold=near_full_threshold)

    #The stock in question has too many empty values
    if near_full_threshold >=30:
        return None, None

    # Fill NaN values with median for columns in near_full_col
    median_values = modified_data[near_full_col].apply(np.nanmedian)
    modified_data[near_full_col] = modified_data[near_full_col].fillna(median_values)

    #Create a copy of the original vector and drop permno and date that are inessential to predict neighboring values
    #Add a time column that essentially acts as a time trend if such trend exist
    full_columns_data = modified_data[near_full_col].copy()
    full_columns_data = full_columns_data.drop(["permno", "yyyymm"], axis=1)
    full_columns_data.loc[:, 'time'] = np.arange(0, len(modified_data))

    #Iterate over the rest columns with missing data
    for col in modified_data.columns:
        if col not in near_full_col:

            y_with_nan = modified_data[col].copy()
            nan_mask = np.isnan(y_with_nan)

            y = y_with_nan[~nan_mask]  # Select non-missing values

            # Add a column of ones to x
            A = np.insert(full_columns_data[~nan_mask].values, 0, 1, axis=1)

            # Find missing values and their indices
            A_pred = np.insert(full_columns_data[nan_mask].values, 0, 1, axis=1)

            # Solve for beta using linear least squares
            assert not np.any(np.isnan(A)), f"NaN values found in column: {col}"
            assert not np.any(np.isnan(y)), f"NaN values found in column: {col}"
            beta, res, rnk, s = lstsq(A, y)

            # Predict missing values
            assert not np.any(np.isnan(A_pred)), f"NaN values found in column: {col}"
            y_pred_missing = np.dot(A_pred, beta)

            # Replace NaN values in y_with_nan with predicted values
            y_with_nan.loc[nan_mask] = y_pred_missing

            # Assign the updated column back to modified_data
            modified_data.loc[:, col] = y_with_nan

    predictors_list = modified_data.columns[modified_data.notnull().all()].tolist()

    return modified_data , predictors_list


def check_stationarity(series):
    # Perform Augmented Dickey-Fuller test
    result = adfuller(series)
    p_value = result[1]
    return p_value <= 0.05

def determine_order(series):
    # Plot ACF and PACF
    plot_acf(series)
    plot_pacf(series)
    # Determine order based on significant peaks or cutoffs
    # For example, look for significant cutoffs in PACF plot
    # Order = number of significant lags in PACF plot
    # Order = number of significant lags in ACF plot if no significant cutoff in PACF
    # Alternatively, you can use information criteria like AIC or BIC to select the order


def Time_series_fitting(serie):
    nan_mask = ~np.isnan(serie)
    nan_indices = np.where(nan_mask)

    first_non_nan_index = nan_indices[0][0]

    serie_nonan = serie[nan_mask]

    if is_constant(serie_nonan):
        return None

    print("First non-NaN index:", first_non_nan_index)
    if not check_stationarity(serie_nonan):
        # Apply differencing if series is not stationary
        series = np.diff(serie)
        # Determine order
    determine_order(serie)
    return None


def is_constant(series, tolerance=1e-6):
    """
    Check if a time series is constant.

    Parameters:
    - series (numpy.ndarray or list): The time series data.
    - tolerance (float): Tolerance level to consider the series as constant.

    Returns:
    - bool: True if the series is constant, False otherwise.
    """
    std_dev = np.std(series)
    return std_dev < tolerance
