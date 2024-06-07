from DataAnalysis.Data_Handler import Load
import numpy as np

def setting_Histogram(data, predictors_list, predictors):
    permno_predictors = Load(predictors_list, show_head=False)

    all_predictors = Load(predictors, show_head=False)

    permno_list = data['permno'].unique()

    all_predictors_list = all_predictors['predictors'].tolist()

    return permno_predictors, permno_list, all_predictors_list

def count_Occurences_Permno(permno_predictors, permno_list, all_predictors_list):
    batons = {predictor: 0 for predictor in all_predictors_list}

    for permno in permno_list:
        associated_predictors = permno_predictors.loc[permno_predictors['permno'] == permno, 'Associated Columns'].iloc[0]
        associated_predictors_list = [predictor.strip()[1:-1] for predictor in associated_predictors.strip('[]').split(',')]

        for predictor in associated_predictors_list:
            batons[predictor] += 1

    return batons

def permno_Occurences(data):
    permno_occurences = data['permno'].value_counts()
    return permno_occurences

def random_predictors(data, number, predictors = "Data/predictors.csv"):
    all_predictors = data.columns
    random_predictors = np.random.choice(all_predictors, number, replace=False)
    return random_predictors

def map_columns_to_angles(predictors_names = "Data/predictors.csv"):
    # Check if the file exists
    if not os.path.isfile(predictors_names):
        raise FileNotFoundError(f"File '{predictors_names}' not found, initialization wasn't complete")

    # Load predictors from CSV file
    predictors_df = Load(predictors_names,show_head=False)

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
