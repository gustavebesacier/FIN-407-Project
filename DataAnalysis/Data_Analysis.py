from DataAnalysis.Data_Handler import Initializer
from DataAnalysis.Data_Handler import Load
from Utils.Data_Treatment import create_groups, delete_repeating_predictors, not_too_empty, highly_corr_predictors
from Utils.Grapher import Histogram_permno_occurrences,Histogram_predictors, plot_3_random_columns, random_nan_repartition
from Utils.Data_Treatment import basic_information, remove_Not_Recurrent_Predictors, average_Filled_Predictors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
import plotly.express as px
import scipy as sp
from scipy.signal import detrend
import csv

def create_subdatasets(data, removed_predictors, deleted_empty_predictors, length_of_subdataset, overlap):
    #The predictors we want to run the code on
    predictors_of_concern = pd.read_csv("Data/PredictorsCleanedGPB_name_only.csv", header=None)

    pred_names = predictors_of_concern[1].tolist()

    start_index = pred_names.index('Accruals') #This is done to avoid the header...

    selected_names = pred_names[start_index:]
    selected_names.insert(0,'yyyymm') #We add back yyyymm for logistics

    all_removed_predictors = list(set(removed_predictors).union(set(deleted_empty_predictors)))

    selected_names = [pred for pred in selected_names if pred not in all_removed_predictors]

    #Take only these ones inside the data
    data = data[selected_names]

    grouped_datas = data.groupby('yyyymm').mean().reset_index() #We group the data by date to make it 2-dimensional
    sub_datasets = []

    start_index = 0
    while start_index + length_of_subdataset <= len(grouped_datas):
        subdataset = grouped_datas.iloc[start_index:start_index + length_of_subdataset]
        #this is only done to compute the correlations, will NOT be used elsewhere
        sd_filled = subdataset.fillna(subdataset.median())
        sub_datasets.append(sd_filled)
        start_index += (length_of_subdataset - overlap)
    
    return selected_names, grouped_datas, sub_datasets

def data_an(data, threshold = 5000):

    #Hardcoded parameters

    #Set the threshold to eliminate not recurrent predictors
    threshold_predictors = threshold
    #The correlation threshold
    threshold_correlations = 0.7
    #How many months there are in 15 years, might want to change this, and the overlap period (33%)
    length_of_subdataset = 15 * 12  
    overlap = 5 * 12  
    threshold_nans = 0.85


    #Make sure this column is in the right format
    data['yyyymm'] = pd.to_datetime(data['yyyymm'], format='%Y%m')

    #Print the information about the raw data
    basic_information(data)

    #Get the NaN-repartition for 3 randoms predictors, plot
    print("Here is a plot of the NaN repartition of 3 randoms predictors")
    random_nan_repartition(data = data, save_path = "Graphs/proportion_of_nans.png")

    #Get the figures for predictor appearance
    print("Here is a histogram on predictor appearance:")
    Histogram_predictors(data)

    #Keep only the predictors that appear in over :threshold_predictors: stocks
    removed_predictors, relevant_predictors = remove_Not_Recurrent_Predictors(data, threshold_predictors)

    print("The predictors removed are:", removed_predictors)

    predictors = relevant_predictors + ["permno"] #We add back permno for logistics
    #Get how many, on average, there are predictors per observation.
    average_Filled_Predictors(data, predictors)

    #Histogram on permno occurrence throughout the data
    Histogram_permno_occurrences(data)

    #We remove the predictors that exhibit over 85% NaN values from the dataset
    max_nans = np.floor(len(data) * threshold_nans).astype(int)
    data, deleted_empty_predictors = not_too_empty(data, max_nans) 

    #We split the data into 9 windows of 15 years with 5 year-overlap
    selected_names, grouped_datas, sub_datasets = create_subdatasets(data = data, removed_predictors = removed_predictors, deleted_empty_predictors = deleted_empty_predictors, length_of_subdataset = length_of_subdataset, overlap = overlap)

    #We create the correlation groups
    groups = []
    for i in range(len(sub_datasets)):
        groups_per_sdf = create_groups(sub_datasets[i])
        groups.append(groups_per_sdf)

    groups_1985_2000 = groups[6]
    groups_1995_2010 = groups[7]
    groups_2005_2020 = groups[8]

    #We print the groups
    print("Here are the correlation groups for the 3 periods of time considered:")
    new_1985 = delete_repeating_predictors(groups = groups_1985_2000, save_path = "Data/groups_1985.txt")
    new_1995 = delete_repeating_predictors(groups = groups_1995_2010, save_path = "Data/groups_1995.txt")
    new_2005 = delete_repeating_predictors(groups = groups_2005_2020, save_path = "Data/groups_2005.txt")

    #We now want to add back the predictors that were not considered as highly correlated with anyone
    not_high_corr_preds_1985 = [pred for pred in selected_names if pred not in highly_corr_predictors(sub_datasets[6])]
    not_high_corr_preds_1995 = [pred for pred in selected_names if pred not in highly_corr_predictors(sub_datasets[7])]
    not_high_corr_preds_2005 = [pred for pred in selected_names if pred not in highly_corr_predictors(sub_datasets[8])]

    not_high_corr_all = [not_high_corr_preds_1985, not_high_corr_preds_1995, not_high_corr_preds_2005]
    
    #We keep only the groups for the time periods that we are interested in, without repeated predictors
    groups = [new_1985, new_1995, new_2005]

    #We add, for each period of time, the "uncorrelated" predictors, so that we have them all stored in the same file
    for i in range(len(not_high_corr_all)):
        for name in not_high_corr_all[i]:
            l = [name]
            groups[i].append(l)

    #Finally, we save it to a .txt
    with open('Data/Groups_With_Uncorrelated_Preds.txt', mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(groups)):
            writer.writerows(groups[i])
            writer.writerow([])

    return groups
