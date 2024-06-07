import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from Utils.Utilities import Predictors_Finder

"""
This files handles data loading as well as building shorter datasets. 
It also runs some data analysis
"""

def Initializer(original_dataframe = "Data/signed_predictors_all_wide_with_returns.csv", short_data_trial = False, shorten_data = "../Data/shorter_data.csv", predictors_names = "Data/predictors.csv",  predictors_per_permno = "Data/permno_predictors.csv"):
    if short_data_trial:
        if not os.path.isfile(shorten_data):
            raise FileNotFoundError(f"File '{shorten_data}' not found, Please load the original set and generate the shorten dataset using Data_shorter")
        data = Load(shorten_data, False)
        print("Data from a shorten dataset has been loaded")

        # Count the number of unique permno values
        num_permno = data['permno'].nunique()
        print("This dataset contains",num_permno, " different permnos")
    else:
        data = Load(original_dataframe, True)
        print("Data from the original file has been loaded")
    if not os.path.isfile(predictors_names):
        Predictors_Finder(data = data, filename = predictors_names)
        print(f"Full predictors list has been generated and saved as {predictors_names}")
    else:
        print(f"predictors list already exists as {predictors_names}")
    if not os.path.isfile(predictors_per_permno):
        Data_Analizer(data=data, filename=predictors_per_permno)
        print(f"Full predictors per permno file has been generated and saved as {predictors_per_permno}")
    else:
        print(f"Predictors per permno file already exists as {predictors_per_permno}")

    return data

def Load(filename = "Data/signed_predictors_all_wide_with_returns.csv", show_head = False):
    """
    Simply loads the file with all the predictors and the returns and share prices
    prints a preview
    :param filename: name of the file containing the data
    :param show_head: if true it will print the first few rows
    :return: return a pandas object
    """
    # Load the dataset
    data = pd.read_csv(filename)

    # Print the first few rows
    if show_head:
        print(data.head())

    return data

def Firm_Extractor(permnos,data):
    """
    Retrieve all rows associated with specific permnos from the DataFrame.
    :param permnos: The ID of each stock
    :param data: Dataframe
    :return: A DataFrame containing all rows associated with the specified permnos.
    """
    # Check if permnos is a single integer or a list
    if isinstance(permnos, int):
        permnos = [permnos]

    # Filter the DataFrame based on the permno
    id_data = data[data['permno'].isin(permnos)]

    return id_data

def Data_Analizer(data, filename = "Data/permno_predictors.csv"):
    """

    :param data:
    :param filename:
    :return:
    """

    # 1. Count the number of observations per stock (permno) and dates (yyyymm)
    obs_per_stock = data.groupby('permno')['yyyymm'].nunique()
    print(obs_per_stock)

    # 2. Plot a density plot for the number of observations per stock

    plt.figure(figsize=(12, 6))
    # sns.kdeplot(obs_per_stock, fill=True, color="r", bw_adjust=0.1)
    plt.plot(obs_per_stock)
    plt.xlabel('Number of years of observations per Stock')
    plt.ylabel('Density')
    plt.title('Density Plot of Number of Observations per Stock')
    plt.show()

    # 3. Run descriptive statistics and identify outliers
    stats = obs_per_stock.describe()
    print(stats)
    print(data.shape)
    total_nans = data.isna().sum().sum()
    print(total_nans)
    num_permno = data['permno'].nunique()
    print(num_permno)
    num_dates = data['yyyymm'].nunique()
    print(num_dates)

    # Init. dict to store associated tables for each permno
    associated_tables = {}
    # This one is for the column names
    associated_column_names = {}

    # Iterate over each permno
    for permno, group in data.groupby("permno"):
        # Exclude the yyyymm and permno column
        wo_permno_date = group.drop(columns=["permno","yyyymm"])

        # for each group find the non-empty columns for this permno
        non_empty_columns = wo_permno_date.columns[wo_permno_date.notnull().any()].tolist()

        # Store associated table for this permno
        associated_tables[permno] = wo_permno_date[non_empty_columns]
        associated_column_names[permno] = non_empty_columns

    # Print associated tables for each permno
    #for permno, table in associated_tables.items():
        #print(f"Permno: {permno}")
        #print(table)
        #print()

    # Print the table with permno and associated non-empty column names
    #print("Permno\tAssociated Columns")
    #for permno, columns in associated_column_names.items():
        #print(f"{permno}\t{', '.join(columns)}")

    permno_predictors = pd.DataFrame(list(associated_column_names.items()), columns=['permno', 'Associated Columns'])

    # Save if needed
    permno_predictors.to_csv(filename, index=False)

    return None

def Data_Shorter(data, nbr_asset, save = False, filename = "Data/shorter_data.csv"):
    """
    The function take the whole dataset and returns a smaller random set of data to work faster.
    If specified, it will save this new dataframe as csv file.
    :param data: original dataset
    :param nbr_asset: number of different permno we want to keep
    :param save: boolean, whether to save the new dataframe as a CSV file
    :param filename: path to save the CSV file
    :return: new dataframe contains random permnos
    """
    random.seed(28)
    permno_unique = data['permno'].unique()

    #Check that the nbr of asset required is doesn't exceed the number of asset
    nbr_asset = min(nbr_asset, len(permno_unique))

    permno_list = random.sample(list(permno_unique), nbr_asset)
    shorter_data = Firm_Extractor(permno_list, data)

    # Save the DataFrame as a CSV file
    if save:
        shorter_data.to_csv(filename, index=False)

    return shorter_data

