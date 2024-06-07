from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector

def aic_bic_hqic(model, preds, output, criterion):
    #Compute/retrieve all the values necessary to compute the values of our criterion of choice
    n = len(output)
    #We include the intercept to get the nb of params (so the +1)
    k = preds.shape[1] + 1 
    fitting = model.predict(sm.add_constant(preds))
    residuals = output - fitting
    rss = np.sum(residuals**2)
    

    if criterion == 'aic':
        return n*np.log(rss/n) + 2*k
    elif criterion == 'bic':
        return n*np.log(rss/n) + np.log(n)*k
    else: 
        return n*np.log(rss/n) + 2* k*np.log(np.log(n))

def aic_bic_hqic_val(preds, output, criterion):
    model = sm.OLS(output, sm.add_constant(preds)).fit()
    return -aic_bic_hqic(model, preds, output, criterion) #just give the value of the criterion for the fitted model

def sffs_met(data):
    empty_model = LinearRegression() #We always start SFFS with an empty model, see report for explanation

    data_to_pred = data['return_lag']
    data_preds = data.drop(columns=['return_lag'])
    
    col = data.columns.to_list()

    scaler = StandardScaler()
    data_preds = scaler.fit_transform(data_preds)

    data_pref__todf = pd.DataFrame(data_preds)

    data_pref__todf.columns = [i for i in col if i != 'return_lag'] #Just to be sure :)

    data_preds = data_pref__todf.copy()
    data_preds = data_preds.drop(columns=["permno", "yyyymm"])

    #Here we should remove est because it is not used anywhere, but generates an error when doing so... gives room for 
    #furthur improvement. 
    sffs_aic = SequentialFeatureSelector(empty_model, 
                                    k_features='best', 
                                    forward=True, 
                                    floating=True, 
                                    scoring=lambda est, X, y: aic_bic_hqic_val(X, y, criterion='aic'), 
                                    cv=0,
                                    n_jobs = -1)

    sffs_bic = SequentialFeatureSelector(empty_model, 
                                    k_features='best', 
                                    forward=True, 
                                    floating=True, 
                                    scoring=lambda est, X, y: aic_bic_hqic_val(X, y, criterion='bic'), 
                                    cv=0,
                                    n_jobs = -1)

    sffs_hqic = SequentialFeatureSelector(empty_model, 
                                    k_features='best', 
                                    forward=True, 
                                    floating=True, 
                                    scoring=lambda est, X, y: aic_bic_hqic_val(X, y, criterion='hqic'),
                                    cv=0,
                                    n_jobs = -1)
    
    #AIC
    sffs_aic.fit(data_preds.values, data_to_pred.values)
    chosen_predictors_AIC = sffs_aic.k_feature_idx_
    names_of_predictors_AIC = data_preds.columns[list(chosen_predictors_AIC)]
    print("The set of predictors that minimize AIC are:", names_of_predictors_AIC)
    underlying_AIC_model = sm.OLS(data_to_pred, sm.add_constant(data_preds[names_of_predictors_AIC])).fit()
    print(underlying_AIC_model.summary())

    #BIC
    sffs_bic.fit(data_preds.values, data_to_pred.values)
    chosen_predictors_BIC = sffs_bic.k_feature_idx_
    names_of_predictors_BIC = data_preds.columns[list(chosen_predictors_BIC)]
    print("The set of predictors that minimize BIC are:", names_of_predictors_BIC)
    underlying_BIC_model = sm.OLS(data_to_pred, sm.add_constant(data_preds[names_of_predictors_BIC])).fit()
    print(underlying_BIC_model.summary())


    #HQIC
    sffs_hqic.fit(data_preds.values, data_to_pred.values)
    chosen_predictors_HQIC = sffs_hqic.k_feature_idx_
    names_of_predictors_HQIC = data_preds.columns[list(chosen_predictors_HQIC)]
    print("The set of predictors that minimize HQIC are:", names_of_predictors_HQIC)
    underlying_HQIC_model = sm.OLS(data_to_pred, sm.add_constant(data_preds[names_of_predictors_HQIC])).fit()
    print(underlying_HQIC_model.summary())