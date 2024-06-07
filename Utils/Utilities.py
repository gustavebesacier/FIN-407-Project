import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def map_columns_to_angles(predictors_names = "Data/predictors.csv"):
    # Check if the file exists
    if not os.path.isfile(predictors_names):
        raise FileNotFoundError(f"File '{predictors_names}' not found, initialization wasn't complete")

    # Load predictors from CSV file
    predictors_df = pd.read_csv(predictors_names)

    # Check if the DataFrame is empty
    if predictors_df.empty:
        raise ValueError(f"The DataFrame '{predictors_names}' is empty.")

    # Check if the 'predictors' column exists
    if 'predictors' not in predictors_df.columns:
        raise KeyError("Column 'predictors' not found in DataFrame.")

    # Extract predictors from the DataFrame
    predictors = predictors_df['predictors']

    num_predictors = len(predictors)
    theta_values = [2 * np.pi * i / num_predictors for i in range(num_predictors)]
    column_angles = {column_name: theta for column_name, theta in zip(predictors, theta_values)}
    return column_angles


def categorical_finder(data, threshold=10):
    """
    Finds the categorical columns, it assumes that categorical columns have less than 10 different values
    :param data: dataframe
    :param threshold: max nbr of different categories
    :return: the name of the categorical columns
    """
    categorical_cols = []
    for col in data.columns:
        if data[col].nunique() <= threshold and data[col].dtype in ['int64', 'float64']:
            categorical_cols.append(col)
    return categorical_cols

def Predictors_Finder(data, filename = "Data/predictors.csv"):
    """
    Saves the name of all existing predictors
    :param data: whole dataset
    :return: None
    """
    if os.path.isfile(filename):
        print(f"File '{filename}' already generated.")
    wo_permno_date = data.drop(columns=["permno","yyyymm"])
    column_names = wo_permno_date.columns.tolist()
    df_predictors = pd.DataFrame(column_names, columns=["predictors"])
    df_predictors.to_csv("Data/predictors.csv", index=False)
    return None

def load_arrays_from_file(file_path):
    """
    This function loads a text file containing a list of estimator
    :param file_path: path of the txt file
    :return: the set of predictors
    """
    with open(file_path, 'r') as file:
        data = file.read()

    # Split the data into array entries
    array_strings = data.strip().split(']), array([')

    # Handle the first and last array edges
    array_strings[0] = array_strings[0].replace('array([', '')
    array_strings[-1] = array_strings[-1].replace(']))', '')

    arrays = []

    for array_str in array_strings:
        # Clean up the string and convert it to a numpy array
        clean_str = array_str.replace('array([', '').replace('])', '')
        array_data = np.fromstring(clean_str, sep=',')
        arrays.append(array_data)

    return arrays

def prepare_lagged_data(data):
    """
    Adds a columns lagged by one time unit
    :param data:
    :return:
    """
    data.sort_values('permno', inplace=True)

    data_lagged = data[data["permno"] == data["permno"].unique()[0]].sort_values(by=['yyyymm'])
    data_lagged['return_lag'] = data_lagged['STreversal'].shift(-1)
    data_lagged.dropna(inplace=True)

    # with alive_bar(len(data.permno.unique()), bar ='halloween') as bar:
    #     for permno in data.permno.unique():
    for permno in tqdm(data.permno.unique()):
        d = data[data["permno"] == permno].sort_values(by=['yyyymm'])
        d['return_lag'] = d['STreversal'].shift(-1)  # the predictors of time t are used to forecast return of time t+1
        d.dropna(inplace=True)  # we loose 1 observation per company (shift)
        data_lagged = pd.concat([data_lagged, d], ignore_index=True,
                                join='outer')  # add the dataframe of this permno to all the others
    return data_lagged

# Function to find the column with the least number of NaNs
def column_with_least_nans(columns, dataframe):
    """
    Finds the column name with least number of nans
    :param columns: list of column names to analyze
    :param dataframe: data
    :return: column name with the least number of nans
    """
    min_nans = float('inf')
    best_column = None
    data_columns = dataframe.columns.tolist()
    for column in columns:
        if column not in data_columns:
            nans = np.inf
        else:
            nans = dataframe[column].isna().sum()
        if nans < min_nans:
            min_nans = nans
            best_column = column
    return best_column

def gen_predictors_for_models(path, data, categorical_columns):
    """
    Quick function to retrieve the predictors found in data analysis
    :param path: path to the predictors file
    :param data: dataframe
    :return: the filtered list
    """
    with open(path, 'r') as file:
        lines = file.readlines()

    # Initialize the lists and current list tracker
    list1, list2, list3 = [], [], []
    current_list = list1

    # Process the lines
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "":
            if current_list == list1:
                current_list = list2
            elif current_list == list2:
                current_list = list3
        else:
            items = stripped_line.split(',')
            if len(items) > 1:
                best_column = column_with_least_nans(items, data)
                if best_column:
                    current_list.append(best_column)
            else:
                current_list.append(items[0])

    # Output the lists
    #print("List1:", list1)
    #print("List2:", list2)
    #print("List3:", list3)

    # Remove categorical columns from the list of predictors as they perform weirdly in the predictors
    filtered_list1 = [acronym for acronym in list1 if acronym not in categorical_columns]

    # Remove categorical columns from the list of predictors as they perform weirdly in the predictors
    filtered_list2 = [acronym for acronym in list2 if acronym not in categorical_columns]

    # Remove categorical columns from the list of predictors as they perform weirdly in the predictors
    filtered_list3 = [acronym for acronym in list3 if acronym not in categorical_columns]

    return filtered_list1, filtered_list2, filtered_list3