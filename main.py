import os
import pandas as pd
from Utils.Utilities import categorical_finder, prepare_lagged_data, gen_predictors_for_models
from DataAnalysis.Data_Handler import Initializer
from DataAnalysis.Data_Analysis import data_an
from DataPreprocessing.Data_fitting import Fit_using_XS_B
from ModellingOptimization.SFFS import sffs
from ModellingOptimization.lasso_elastic import full_run
from ModellingOptimization.NonParametrical import NPE, rank_transform
from Predicting.FLASH import inference, train_model

short_trial = True
data_analysis = False
data_preprocessing = False
sffs_run = False
lasso_enet = False
non_param = False
predict = True
save = False

def main():
    # Either loads the original full dataset, can crash on certain computers
    if short_trial:
        #Loads a datafram with 5000 different permnos selected at random
        data_raw = Initializer(short_data_trial=True, shorten_data="Data/short_5000bis.csv")
    else :
        data_raw = Initializer(short_data_trial=False)

    # Run basic analysis to get information on the data
    if data_analysis:
        #If you want tu run with a short trial
        if short_trial:
            groups = data_an(data_raw, threshold=200)
            print(groups)
        else:
            groups = data_an(data_raw)
            print(groups)

    # If true it will generate a new dataframe with nan values fitted using cross-correlation and time serie
    if data_preprocessing:
        predictors_shorten = pd.read_csv("Data/PredictorsCleanedGPB_name_only.csv")["Acronym"].tolist()
        categorical_columns = categorical_finder(data_raw)

        # Remove categorical columns from the list of predictors as they perform weirdly in the predictors
        predictors_shorten.append("")
        filtered_predictors = [acronym for acronym in predictors_shorten if acronym not in categorical_columns]
        nbr_predictors = len(filtered_predictors)

        # Print the length of the new list
        #print("Number of predictors used for data fitting:", nbr_predictors)

        filtered_predictors.append("permno")
        filtered_predictors.append("yyyymm")
        #filtered_predictors.append("Size")
        #filtered_predictors.append("Price")
        #filtered_predictors.append("STreversal")
        #nbr_predictors += 3

        # As described in the report we chose to analyse section of 15 years
        start_date = 200501
        end_date = 202001

        # shorten the data to fit, it can also be done on the whole data, we cut the data to shorten the length of running
        data = data_raw[(data_raw["yyyymm"] >= start_date) & (data_raw["yyyymm"] <= end_date)]

        # Restrict the number of fitting predictors
        filtered_data = data[filtered_predictors]

        # Run the data fitting
        Nbr_factors = 45
        regularization_factor = 0.0001
        data_fitted = Fit_using_XS_B(filtered_data, K_factors=Nbr_factors, nbr_predictors=nbr_predictors,
                                     gamma=regularization_factor)

        #From here we focus to build the dataset for machine learning
        filtered_list_198501_200001, filtered_list_199501_201001, filtered_list_200501_202001 = gen_predictors_for_models(path = "Data/Uncorrelated_Preds.txt", data= filtered_data, categorical_columns= categorical_columns)

        if start_date == 198501:
            filtered_list_198501_200001.append("permno")
            filtered_list_198501_200001.append("yyyymm")
            # Add the target (the y we want to predict)
            pre_lagged_filter = data_fitted[filtered_list_198501_200001]
            data_lagged = prepare_lagged_data(pre_lagged_filter)

            #Save the file so you don't need to build it everytime
            if save:
                if short_trial:
                    data_lagged.to_csv("Data/short_final_data_198501_200001.csv", index=False)
                else:
                    data_lagged.to_csv("Data/final_data_198501_200001.csv", index=False)

        if start_date == 199501:
            filtered_list_199501_201001.append("permno")
            filtered_list_199501_201001.append("yyyymm")
            # Add the target (the y we want to predict)
            pre_lagged_filter = data_fitted[filtered_list_199501_201001]
            data_lagged = prepare_lagged_data(pre_lagged_filter)

            #Save the file so you don't need to build it everytime
            if save:
                if short_trial:
                    data_lagged.to_csv("Data/short_final_data_199501_201001.csv", index=False)
                else:
                    data_lagged.to_csv("Data/final_data_199501_201001.csv", index=False)

        if start_date == 200501:
            filtered_list_200501_202001.append("permno")
            filtered_list_200501_202001.append("yyyymm")
            # Add the target (the y we want to predict)
            pre_lagged_filter = data_fitted[filtered_list_200501_202001]
            data_lagged = prepare_lagged_data(pre_lagged_filter)

            #Save the file so you don't need to build it everytime
            if save:
                if short_trial:
                    data_lagged.to_csv("Data/short_final_data_200501_202001.csv", index=False)
                else:
                    data_lagged.to_csv("Data/final_data_200501_202001.csv", index=False)

    else:
        print("too bad, you will have nans :(")

    # This part runs the Sequential Forward Floating Selection estimation for the predictors
    if sffs_run:
        data = pd.read_csv("Data/final_data_199501_201001.csv")
        sffs(data)
        print("Agu")

    if lasso_enet:
        full_run(test=False)

    # This part runs the non parametric estimation for the predictors
    if non_param:
        if not data_preprocessing:
            file_path = "Data/short_final_data_199501_201001.csv"
            assert os.path.exists(file_path), f"File not found: {file_path}, please generate it by changing data_preprocessing to True, if you are running the short process, ensure the short files have been created"
            data_lagged = Initializer(short_data_trial=True, shorten_data=file_path)

        #Use the rank tranformation described in the course
        data_transformed = rank_transform(data_lagged)

        y = data_transformed[["yyyymm", "return_lag"]].copy()
        X = data_transformed.drop("return_lag", axis=1)
        L = 2 #Number of split for the splines; minimum number is one

        beta_list, mse_list, r2_list = NPE(X=X, y=y, lbd1=0.675, lbd2=0.23, L=L, threshold=1e-8)

        print(mse_list)
        print(r2_list)
        print(beta_list)

    if predict:
        for path in ["Data/Train_200501_202001.csv"]:
            train_model(
                path_data=path,
                epochs=75
            )

        for path in ["Data/Test_202001_202206.csv"]:
            inference(
                path_data=path,
                show=True
            )
        print("Welcome to the realm of FLASH")

if __name__ == '__main__':
    main()
