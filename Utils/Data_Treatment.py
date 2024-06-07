import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

from Utils.shared_Treatment_Grapher import setting_Histogram, count_Occurences_Permno
from itertools import combinations
from collections import Counter

def basic_information(data):
    #How many years of data we have
    years = len((data['yyyymm'].dt.year).unique())
    print(f"There are {years} years of data")
    #How many different stocks
    print("There are",  len(data['permno'].unique()), "different stocks")
    #Shape of the data
    print("The shape of the data is", data.shape)
    #How many NaNs
    print("There are", data.isna().sum().sum(), "NaNs")


def remove_Not_Recurrent_Predictors(data, threshold, predictors_list="Data/permno_predictors.csv", predictors="Data/predictors.csv", save_path="Data/Removed_predictors_threshold_appearance.csv"):
    permno_predictors, permno_list, all_predictors_list = setting_Histogram(data, predictors_list, predictors)
    
    batons = count_Occurences_Permno(permno_predictors, permno_list, all_predictors_list)

    #Keep only the predictors that appear in *threshold* or more stocks (filtered by permno)
    cropped_batons = {predictor: count for predictor, count in batons.items() if count >= threshold}
    removed_predictors = {predictor: count for predictor, count in batons.items() if count < threshold}

    #Keep only the relevant predictors (will be useful later)
    relevant_predictors = list(cropped_batons.keys())

    return removed_predictors, relevant_predictors

def average_Filled_Predictors(data, predictors): 
    data = data[predictors] 
    #How many predictors are available (in mean) for each pernmo
    filled_predictors_per_permno = data.groupby('permno').apply(lambda x: x[predictors].notna().sum().mean())
    #Overall average
    average_filled_predictors = filled_predictors_per_permno.mean()

    print(f"There are on average {average_filled_predictors} filled predictors per permno")


#For the Cluster Analysis

def highly_corr_predictors(data, threshold = 0.7):
    #Get correlation matrix for the data
    corr_matrix = corr_mat(data)
    #We do this to avoid adding all the columns to the list (!)
    np.fill_diagonal(corr_matrix.values, 0)
    #If two predictors are highly correlated, their names are added to the list.
    names_pairwise_corr = [column for column in corr_matrix.columns if any(corr_matrix[column] >= threshold)]
    names_pairwise_corr = list(set(names_pairwise_corr))

    return names_pairwise_corr

#Correlation matrix (Pearson)
def corr_mat(data):
    return data.corr()

#Correlation between 2 predictors
def pairwise_correlation(corr_matrix, asset1, asset2):
    return corr_matrix.loc[asset1, asset2]


def create_groups(data, t=0.7):
    """
    Creates highly correlated groups (clusters).

    :param data: data DataFrame
    :param t: correlation threshold 
    :temp correlated_preds: list of predictors that are highly correlated
    :return all_groups: list of groups
    """

    correlated_preds = highly_corr_predictors(data) 
    corr_matrix = corr_mat(data)

    #Initialization
    all_groups = []

    #We take initally all pairs of predictors that are highly correlated
    #A precaution we took was to recheck that the pair we were considering
    #was indeed highly correlated, as :correlated_preds: contains only names.
    initial_groups = [
        [pair[0], pair[1]] for pair in combinations(correlated_preds, 2)
        if np.abs(pairwise_correlation(corr_matrix, *pair)) >= t 
    ]

    for correlated_pair in initial_groups:
        group = set(correlated_pair)
        predictor_was_added = True

        while predictor_was_added:
            predictor_was_added = False
            for predictor in correlated_preds:
                if predictor not in group:
                    if all(np.abs(pairwise_correlation(corr_matrix, predictor, predictor_already_there)) >= t for predictor_already_there in group):
                        group.add(predictor)
                        predictor_was_added = True

        all_groups.append(list(group))

    #Remove duplicate groups
    all_groups = [list(group) for group in set(frozenset(group) for group in all_groups)]

    return all_groups

def delete_repeating_predictors(groups, save_path):
    all_predictors = [predictor for group in groups for predictor in group]
    predictor_appearances = Counter(all_predictors)

    #Check that the predictor appears at least twice
    repeating_predictors = {predictor for predictor, count in predictor_appearances.items() if count >= 2} 
    #We do the new groups now removing all the predictors that appear more than twice, and we iterate over the predictors of a group and over all the groups. 
    new_groups = [
        [predictor for predictor in group if predictor not in repeating_predictors]
        for group in groups
    ]
    
    new_groups = [group for group in new_groups if group]
    
    for i, group in enumerate(new_groups):
        print(f"Group {i+1}: {', '.join(group)}")
    
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_groups)
    
    return new_groups

def not_too_empty(data, max_nans): 

    deleted_preds = [pred for pred in data.columns if data[pred].isna().sum() > max_nans]
    data_update = data.drop(columns = deleted_preds)
    
    return data_update, deleted_preds
