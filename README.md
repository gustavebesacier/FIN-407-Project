# Decoding Future Stock Returns: Revealing the Key Predictors

This is a project from Guillaume Ferrer, Gustave Paul Besacier, Eva Perazzi and Agustina Maria Zein, students from the course Machine Learning in Finance FIN-407 from the Swiss Federal Institute of Technology Lausanne (EPFL). 

## About the project 📈📊

- Goal: Retrieve a subset of predictors that best predict the stocks returns.

- The code is accompanied by a 20-page report, which is available in the repository (...github).


## Installation 💻
The code is optimized for Python 3.11.


### Library
The following library are used:
- Numpy
- Matplotlib
- Scikit-Learn
- Pandas
- Scipy
- Json
- Os
- seaborn
- tqdm
- torch
- transformers
- mlxtend
- (IPython)

## Files 📁

### Main files
- [main.py](main.py) : Master file, desired processes are called from it. 

## Directories

### [Data](Data)
Contains the data used in the project in different formats(.txt, .csv).
Additionally some contains shorten dataset computed for shortened run times. The required data to run the code is available here https://drive.google.com/drive/folders/1ivoX5Kiannv-GN9mML8K0n7tHz3baE38?usp=share_link.

### [DataAnalysis](DataAnalysis)
Contains the various files to handle the data and run statistical analysis
- [Data_Handler.py](DataAnalysis%2FData_Handler.py) : Handles data loading as well as creating shorter datasets. Can run small analysis
- [Data_Analysis.py](DataAnalysis%2Data_Analysis.py) : Does the data analysis when called in main.

### [DataPreprocessing](DataPreprocessing) 
Handles the various techniques to fit nan values and creates the dataset to run the machine learning algorithms
- [Data_fitting.py](DataPreprocessing%2FData_fitting.py) : Imputes the missing values using the cross-sectional and time series information from the data.
- [FittingTries.py](DataPreprocessing%2FFittingTries.py) : Old imputation tries, it runs firm independent fitting.

### [ModellingOptimization](ModellingOptimization)
This directory is composed of all the methods we have used to find a predictors subset
- [NonParametrical.py](ModellingOptimization%2FNonParametrical.py) : This file contains multiple implementation of the non-parametrical estimation using the adaptive group Lasso.
- [lasso_elastic.py](ModellingOptimization%2lasso_elastic.py) : This file contains the implementation of Lasso models and bootstrapped-enhanced technique based on cross validation and SURE framework optimal regularization parameter determination. It also contains these methods applied to Elastic-Net.
- [SFFS.py](ModellingOptimization%2SFFS.py) : Contains the implementation of the SFFS (Sequential Forward Feature Selection) using AIC, BIC and HQIC criteria.

### [Predicting](Predicting)
Contains the implementation of the deep neural network using the predcitors we found to assess them.
- [FLASH.py](Predicting%2FLASH.py) : this module implements FAST (Financial Learning Algorithm for Signal Heuristics), a deep neural network designed to predict return direction and therefore provide trading signal (buy/sell). The module contains all the data handling, model training and model inference.

### [Graphs](Graphs)
This directory contains many different graph we used to analyze or represent data.

### [Utils](Utils)
Contains various utility files.
- [Grapher.py](Utils%2FGrapher.py) : Contains multiple functions to plot various graphs
- [Utilities.py](Utils%2FUtilities.py) : Functions with single use
- [shared_Treatment_Grapher.py](Utils%2shared_Treatment_Grapher.py) : Contains the methods shared by more than one .py file to avoid circular imports
- [Data_Treatment.py](Utils%2Data_Treatment.py) : Contains function done to analyse the data, such as to give the basic information about the rax data, the different treatments we applied to the data, the group creation and handling, etc. 

### [Archive](Archive)
Old files, they should not be used as they may contain inefficient code or even errors.

## Usage 🫳
The code can be downloaded on the GitHub repository. Usage is of a standard Python code.
The original file is quite large and it is available in a Google Drive: https://drive.google.com/drive/folders/1ivoX5Kiannv-GN9mML8K0n7tHz3baE38?usp=share_link

## Contact 📒
- Guillaume Ferrer: guillaume[dot]ferrer[at]epfl[dot]ch
- Gustave Paul Besacier: gustave[dot]besacier[at]epfl[dot]ch
- Eva Perazzi: eva[dot]perazzi[at]epfl[dot]ch
- Agustina Maria Zein: agustina[dot]zein[at]epfl[dot]ch


Project link: [https://github.com/gustavebesacier/MouLa](https://github.com/donQuiote/FIN-407-Project.git)

## Acknowledgments 🤗
We thank FIN-407 team for their support.
