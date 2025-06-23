#!/usr/bin/env python
# coding: utf-8

import pyreadr as pr
import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import skew, kurtosis, pearsonr 
import statistics # for computing the mode of a variable
import platform   # to print the version of Python and selected libraries
import openpyxl
import pyreadstat
import matplotlib.image as mpimg

# imputation data
import miceforest as mf
from pyampute.exploration.mcar_statistical_tests import MCARTest

# Libraries for Machine Learning
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import r2_score, make_scorer, mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import shap
from skopt import BayesSearchCV # for Bayesian Optimization
from skopt.space import Real, Categorical, Integer
import xgboost as xgb
from xgboost import XGBRegressor
from joblib import dump,load

# import Custom Functions from function.py
from function import *
from ML_function import *

def Train_Test_normal (my_data,n,n1):
    my_df = my_data.values
    my_X = my_df[:n,1:n1]
    my_y = my_df[:n,0]
    print('Input shape:', my_X.shape)
    print('Output shape:', my_y.shape)
    my_X_train, my_X_test, my_y_train, my_y_test = train_test_split(my_X, my_y, test_size = 0.25, random_state = 1234)
    scaler  = StandardScaler()
    my_X_train = scaler.fit_transform(my_X_train)
    my_X_test  = scaler.transform(my_X_test)
    print(my_X_train.shape)  # features (X) for training data
    print(my_X_test.shape)   # features (X) for test data
    print(my_y_train.shape)  # target/output (Y) for training data
    print(my_y_test.shape)
    print('-'*40)
    # print("TRAIN")
    # print(round(pd.DataFrame(my_X_train).describe(), 3))
    # print("TEST")
    # print(round(pd.DataFrame(my_X_test).describe(), 3))
    return my_X_train, my_X_test, my_y_train, my_y_test 



def eNET_function(my_enet_hyparam, my_enet_X_train, my_enet_X_test, my_enet_y_train, my_enet_y_test, my_features, my_dir):
    print('-'*100)
    print('-'*100)
    print('-'*100)
    results = {}
    my_elastic_r_grid_search = GridSearchCV(ElasticNet(max_iter=400000),
                                      my_enet_hyparam,
                                      cv=10)
    my_elastic_r_grid_search.fit(my_enet_X_train, my_enet_y_train)

    # Store best score and hyperparameters for this k value
     
    results = {
         'score': my_elastic_r_grid_search.best_score_,
         'params': my_elastic_r_grid_search.best_params_
     }
    
     # Display the results
    
    print(f'Best hyperparameters (k = {results}):', results['params'])
    print('-'*40)
    
    pd.DataFrame.from_dict(my_elastic_r_grid_search.cv_results_) # Inspect column `mean_test_score`, in order to manually verify the presence of
                                                          #   significant excursions on the whole range
                                                          #   of maximum values of best scores obtained by changing the hyperparameters
    print(
    pd.DataFrame.from_dict(
       my_elastic_r_grid_search.cv_results_
    )['mean_test_score'].describe()
    )
    print('-'*40)
    # Hypeparameters from the best cross-validation
    my_elastic_tuned = ElasticNet(
    alpha   =  results["params"]["alpha"], 
    l1_ratio = results["params"]["l1_ratio"]
    )
    my_elastic_tuned.fit(my_enet_X_train, my_enet_y_train)

    my_predict_elNet_tuned_train = my_elastic_tuned.predict(my_enet_X_train)
  
    
    
    # Prediction on test set
    my_predict_elNet_tuned_test = my_elastic_tuned.predict(my_enet_X_test)
    
    # Train (fit) model and run prediction on train and test data
    # Compute MAE for the training set (useful to subsequently check overfitting)
    
    my_mae_elast_net_train = mean_absolute_error(my_enet_y_train, my_predict_elNet_tuned_train)
    print(f"MAE for train set (Elastic Net): {my_mae_elast_net_train:.3f}")
    
    # Compute MAE for test set
    
    my_mae_elast_net_test = mean_absolute_error(my_enet_y_test, my_predict_elNet_tuned_test)
    print(f"MAE for test set (Elastic Net): {my_mae_elast_net_test:.3f}")
    
    
    # Compute R^2 for train set (useful to subsequently check overfitting)
    
    my_r2_elast_net_train = r2_score(my_enet_y_train, my_predict_elNet_tuned_train)
    print(f"R^2 for train set (Elastic Net): {my_r2_elast_net_train:.3f}")
    
    # Compute R^2 for test set
    
    my_r2_elast_net_test = r2_score(my_enet_y_test, my_predict_elNet_tuned_test)
    print(f"R^2 for test set (Elastic Net): {my_r2_elast_net_test:.3f}")


    # Prediction on train set (useful to subsequently check overfitting)
    
    
    print('-'*40)
    subset_ratio = 0.25
    n_repeats    = 1000
    scaler  = StandardScaler()
    print_values=True
    MAE_train = []
    MAE_test  = []
    R2_train  = []
    R2_test   = []

    n_train = my_enet_X_train.shape[0]
    n_test  = my_enet_X_test.shape[0]

    # fix a seed
    np.random.seed(1234)

    #repeat 'n_repeats' times
    for i in range(n_repeats):
        
        #select a subset
        idx_train = np.random.choice(n_train, int(subset_ratio*n_train))
        idx_test  = np.random.choice(n_test,  int(subset_ratio*n_test ))

        # Use a scaler for train and test partition
        X_train_ = scaler.fit_transform(my_enet_X_train[idx_train])
        X_test_  = scaler.transform(my_enet_X_test[idx_test])
        
        #generate prediction on the subset
        y_pred_train_ = my_elastic_tuned.predict(X_train_)
        y_pred_test_  =  my_elastic_tuned.predict(X_test_)

        #compute (and store) performances on the subset
        ## for MAE
        MAE_train.append(mean_absolute_error(my_enet_y_train[idx_train], y_pred_train_))
        MAE_test.append(mean_absolute_error(my_enet_y_test[idx_test], y_pred_test_))
        ## for R_squared
        R2_train.append(r2_score(my_enet_y_train[idx_train], y_pred_train_))
        R2_test.append(r2_score(my_enet_y_test[idx_test], y_pred_test_))    

    train_median_MAE, train_CI5_MAE, train_CI95_MAE = np.quantile(MAE_train, [0.5, 0.05, 0.95])
    test_median_MAE, test_CI5_MAE, test_CI95_MAE    = np.quantile(MAE_test, [0.5, 0.05, 0.95])
    train_median_r2, train_CI5_r2, train_CI95_r2 = np.quantile(R2_train, [0.5, 0.05, 0.95])
    test_median_r2, test_CI5_r2, test_CI95_r2    = np.quantile(R2_test, [0.5, 0.05, 0.95])    
    
    
    if print_values:
        print(f"MAE train (Elastic Net): {train_median_MAE:.3f} [{train_CI5_MAE:.3f}-{train_CI95_MAE:.3f}]")
        print(f"MAE test (Elastic Net): {test_median_MAE:.3f} [{test_CI5_MAE:.3f}-{test_CI95_MAE:.3f}]")
        print(f"R^2 train (Elastic Net): {train_median_r2:.3f} [{train_CI5_r2:.3f}-{train_CI95_r2:.3f}]")
        print(f"R^2 test (Elastic Net): {test_median_r2:.3f} [{test_CI5_r2:.3f}-{test_CI95_r2:.3f}]")

    my_ove = {
    'Alghorithm': ["Elastic Net"] * 2,
    'Partition': ['Train', 'Test'],
    'MAE Median': [train_median_MAE,test_median_MAE],
    'MAE CI 5%': [train_CI5_MAE, test_CI5_MAE],
    'MAE CI 95%': [train_CI95_MAE, test_CI95_MAE],
    'R^2 Median': [train_median_r2, test_median_r2],
    'R^2 CI 5%': [train_CI5_r2, test_CI5_r2],
    'R^2 CI 95%': [train_CI95_r2, test_CI95_r2]
    }
    my_ove = pd.DataFrame(my_ove)
  
    
    my_ove.insert(loc=2, column='MAE', value=[my_mae_elast_net_train,my_mae_elast_net_test])
    my_ove.insert(loc=2, column='R^2', value=[my_r2_elast_net_train,my_r2_elast_net_test])
    
    print('-'*40)
    print(f'b0 (estimated): {my_elastic_tuned.intercept_}')
    my_param_1 = pd.DataFrame(
            {
                'Features': my_features,
                'Parameters':  my_elastic_tuned.coef_
            }
    ).round(3)
    my_param_1= my_param_1.sort_values(by='Parameters', ascending=False)
    
    print(my_param_1)


    
    # Assuming 'elastic_tuned.coef_' and 'features' are already defined
    coef = my_elastic_tuned.coef_
    df_coef = pd.DataFrame({'feature': my_features, 'coef': coef})
    df_coef['abs_coef'] = df_coef['coef'].abs()
    df_coef = df_coef.sort_values('abs_coef', ascending=False)
    
    # Define colors based on the original coefficient value
    colors = ['red' if c < 0 else 'blue' for c in df_coef['coef']]
    
    # Create the bar plot with absolute values
    
    plt.bar(df_coef['feature'], df_coef['abs_coef'], color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Absolute Coefficient")
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig(f'{my_dir}Feature_importance.png')
    plt.show()
    
    return(my_ove)




def rf_function(my_rf_hyparam, my_rf_X_train, my_rf_X_test, my_rf_y_train, my_rf_y_test, my_features, my_dir):
    print('-'*100)
    print('-'*100)
    print('-'*100)
    # Create a Random Forest regressor
    my_rf = RandomForestRegressor(random_state = 1234)

    # Perform Bayesian Optimization with 5-fold cross-validation
    my_opt = BayesSearchCV(my_rf,
                        my_rf_hyparam,
                        n_iter=100,
                        cv=5,
                        random_state=1234,
                        n_jobs=-1)
    
    # Fit the optimization object to the data
    my_opt.fit(my_rf_X_train, my_rf_y_train)
    
    # Print the best hyperparameters found
    print("Best hyperparameters:", my_opt.best_params_)
    h = my_opt.best_params_
    # Perform cross-validation with the best hyperparameters
    np.random.seed(1234)
    cv_results = cross_val_score(my_opt.best_estimator_,
                                 my_rf_X_train,
                                 my_rf_y_train,
                                 cv=5)
    
    # Print the cross-validation scores
    print("Cross-validation scores:", cv_results)
    print("Mean cross-validation score:", np.mean(cv_results))
    print('-'*40)
    # Add specified hyperparameters (manually)
    
    my_rf_tuned = RandomForestRegressor(bootstrap = h["bootstrap"],
                                     max_depth = h["max_depth"],
                                     max_features = h["max_features"],
                                     min_samples_leaf = h["min_samples_leaf"],
                                     min_samples_split = h["min_samples_split"],
                                     n_estimators = h["n_estimators"],
                                     random_state=1234)
    
    # Training
    my_rf_tuned.fit(my_rf_X_train, my_rf_y_train)
    
    # Prediction on train set (useful to subsequently check overfitting)
    predict_rf_tuned_train = my_rf_tuned.predict(my_rf_X_train)
    
    # Prediction on test set
    predict_rf_tuned_test = my_rf_tuned.predict(my_rf_X_test)
    
    # Compute MAE for the training set (useful to subsequently check overfitting)
    
    mae_rf_train = mean_absolute_error(my_rf_y_train, predict_rf_tuned_train)
    print(f"MAE for train set (Random Forest): {mae_rf_train:.3f}")
    
    # Compute MAE for test set
    
    mae_rf_test = mean_absolute_error(my_rf_y_test, predict_rf_tuned_test)
    print(f"MAE for test set (Random Forest): {mae_rf_test:.3f}")
    
    
    # Compute R^2 for train set (useful to subsequently check overfitting)
    
    r2_rf_train = r2_score(my_rf_y_train, predict_rf_tuned_train)
    print(f"R^2 for train set (Random Forest): {r2_rf_train:.3f}")
    
    # Compute R^2 for test set
    
    r2_rf_test = r2_score(my_rf_y_test, predict_rf_tuned_test)
    print(f"R^2 for test set (Random Forest): {r2_rf_test:.3f}")



    print('-'*40)
    subset_ratio = 0.25
    n_repeats    = 1000
    scaler  = StandardScaler()
    print_values=True
    MAE_train = []
    MAE_test  = []
    R2_train  = []
    R2_test   = []

    n_train = my_rf_X_train.shape[0]
    n_test  = my_rf_X_test.shape[0]

    # fix a seed
    np.random.seed(1234)

    #repeat 'n_repeats' times
    for i in range(n_repeats):
        
        #select a subset
        idx_train = np.random.choice(n_train, int(subset_ratio*n_train))
        idx_test  = np.random.choice(n_test,  int(subset_ratio*n_test ))

        # Use a scaler for train and test partition
        X_train_ = scaler.fit_transform(my_rf_X_train[idx_train])
        X_test_  = scaler.transform(my_rf_X_test[idx_test])
        
        #generate prediction on the subset
        y_pred_train_ = my_rf_tuned.predict(X_train_)
        y_pred_test_  =  my_rf_tuned.predict(X_test_)

        #compute (and store) performances on the subset
        ## for MAE
        MAE_train.append(mean_absolute_error(my_rf_y_train[idx_train], y_pred_train_))
        MAE_test.append(mean_absolute_error(my_rf_y_test[idx_test], y_pred_test_))
        ## for R_squared
        R2_train.append(r2_score(my_rf_y_train[idx_train], y_pred_train_))
        R2_test.append(r2_score(my_rf_y_test[idx_test], y_pred_test_))    

    train_median_MAE, train_CI5_MAE, train_CI95_MAE = np.quantile(MAE_train, [0.5, 0.05, 0.95])
    test_median_MAE, test_CI5_MAE, test_CI95_MAE    = np.quantile(MAE_test, [0.5, 0.05, 0.95])
    train_median_r2, train_CI5_r2, train_CI95_r2 = np.quantile(R2_train, [0.5, 0.05, 0.95])
    test_median_r2, test_CI5_r2, test_CI95_r2    = np.quantile(R2_test, [0.5, 0.05, 0.95])    
    
    
    if print_values:
        print(f"MAE train (Random Forests): {train_median_MAE:.3f} [{train_CI5_MAE:.3f}-{train_CI95_MAE:.3f}]")
        print(f"MAE test (Random Forests): {test_median_MAE:.3f} [{test_CI5_MAE:.3f}-{test_CI95_MAE:.3f}]")
        print(f"R^2 train (Random Forests): {train_median_r2:.3f} [{train_CI5_r2:.3f}-{train_CI95_r2:.3f}]")
        print(f"R^2 test (Random Forests): {test_median_r2:.3f} [{test_CI5_r2:.3f}-{test_CI95_r2:.3f}]")

    my_ove = {
    'Alghorithm': ["Random Forests"] * 2,
    'Partition': ['Train', 'Test'],
    'MAE Median': [train_median_MAE,test_median_MAE],
    'MAE CI 5%': [train_CI5_MAE, test_CI5_MAE],
    'MAE CI 95%': [train_CI95_MAE, test_CI95_MAE],
    'R^2 Median': [train_median_r2, test_median_r2],
    'R^2 CI 5%': [train_CI5_r2, test_CI5_r2],
    'R^2 CI 95%': [train_CI95_r2, test_CI95_r2]
    }
    my_ove = pd.DataFrame(my_ove)

    my_ove.insert(loc=2, column='MAE', value=[mae_rf_train,mae_rf_test])
    my_ove.insert(loc=2, column='R^2', value=[r2_rf_train,r2_rf_test])
    
    Data_2 = pd.DataFrame(
        {
            'Features': my_features,
            'Parameters': my_rf_tuned.feature_importances_
        }
    ).round(3)
    Data_2= Data_2.sort_values(by='Parameters', ascending=False)
    # Data_2.to_excel(dir_4, index=False)
    print(Data_2)

    
    coef = my_rf_tuned.feature_importances_ # in random forest we no longer have `.coef_`, but `.feature_importances_`
    df_coef = pd.DataFrame({'feature': my_features, 'coef': coef})
    df_coef = df_coef.sort_values('coef', ascending=False)
    plt.figure()  # Create a new figure
    plt.bar(df_coef['feature'], df_coef['coef'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Coefficient")
    plt.tight_layout()
    plt.savefig(f'{my_dir}Feature_importante(Random_Forests).png')
    plt.show()

    plt.figure()  # Create a new figure
    # Fits the explainer (i.e., explain the model's predictions using SHAP)
    my_explainer_rf = shap.Explainer(my_rf_tuned.predict, my_rf_X_test)
    
    # Calculates the SHAP values
    shap_values_rf = my_explainer_rf(my_rf_X_test)
    
    # Set the figure size
    plt.figure(figsize=(4, 3))
    
    # Violin plot
    shap.summary_plot(shap_values_rf,
                      # plot_type='violin',
                      feature_names = my_features,
                      show=False)                       # NOTE: if `show = True`, then you cannot export the figure.
                                                        #       Also avoid `plt.show()` for the same reason
    # Export figure in png with 300 dpi
    plt.savefig(f'{my_dir}Shap_(Random_Forests).png',
        dpi=300,
        bbox_inches='tight' # bbox_inches='tight' ensures that the entire figure is included in the saved image without any cropping
    )


    return(my_ove)

    
# ove_2 = overf_check(rf_tuned,
#                 title_overfit = 'Overfitting check - Random Forest',
#                 my_dpi = 100,
#                 print_values=True,
#                 model_name= 'Random Forest')


# print('-'*40)

def XGB_function(my_xgb_hyparam, my_xgb_X_train, my_xgb_X_test, my_xgb_y_train, my_xgb_y_test, my_features, my_dir):
    print('-'*100)
    print('-'*100)
    print('-'*100)
    
    my_model_xgb = XGBRegressor()

    my_xgb_opt = BayesSearchCV(
        my_model_xgb,
        my_xgb_hyparam,
        n_iter=100,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=1234,
        verbose=0,
        n_jobs=-1
    )

    my_xgb_opt.fit(my_xgb_X_train, my_xgb_y_train)
    
    best = my_xgb_opt.best_estimator_
    
    print("Best hyperparameters:", my_xgb_opt.best_params_)
    h_xgb = my_xgb_opt.best_params_
    
    # Perform cross-validation with the best hyperparameters
    np.random.seed(1234)
    my_xgb_cv_results = cross_val_score(
        my_xgb_opt.best_estimator_,
        my_xgb_X_train,
        my_xgb_y_train,
        cv=5
    )
    print(my_xgb_cv_results)
    
    # Add specified hyperparameters 
    
    my_xgb_tuned = xgb.XGBRegressor(
        colsample_bylevel = h_xgb["colsample_bylevel"],
        colsample_bytree = h_xgb["colsample_bytree"],
        gamma = h_xgb["gamma"],
        learning_rate = h_xgb["learning_rate"],
        max_depth = h_xgb["max_depth"],
        min_child_weight = h_xgb["min_child_weight"],
        n_estimators = h_xgb["n_estimators"],
        reg_alpha = h_xgb["reg_alpha"],
        reg_lambda = h_xgb["reg_lambda"],
        subsample = h_xgb["subsample"],
        random_state=1234
    )
    
    # Training
    my_xgb_tuned.fit(my_xgb_X_train, my_xgb_y_train)
    
    # Prediction on train set (useful to subsequently check overfitting)
    predict_xgb_tuned_train = my_xgb_tuned.predict(my_xgb_X_train)
    
    # Prediction on test set
    predict_xgb_tuned_test = my_xgb_tuned.predict(my_xgb_X_test)
    
    
    print('-'*40)
    # Compute MAE for the training set (useful to subsequently check overfitting)
    
    mae_xgb_train = mean_absolute_error(my_xgb_y_train, predict_xgb_tuned_train)
    print(f"MAE for train set (xgb): {mae_xgb_train:.3f}")
    
    # Compute MAE for test set
    
    mae_xgb_test = mean_absolute_error(my_xgb_y_test, predict_xgb_tuned_test)
    print(f"MAE for test set (xgb): {mae_xgb_test:.3f}")
    
    
    # Compute R^2 for train set (useful to subsequently check overfitting)
    
    r2_xgb_train = r2_score(my_xgb_y_train, predict_xgb_tuned_train)
    print(f"R^2 for train set (xgb): {r2_xgb_train:.3f}")
    
    # Compute R^2 for test set
    
    r2_xgb_test = r2_score(my_xgb_y_test, predict_xgb_tuned_test)
    print(f"R^2 for test set (xgb): {r2_xgb_test:.3f}")
    print('-'*40)


    print('-'*40)
    subset_ratio = 0.25
    n_repeats    = 1000
    scaler  = StandardScaler()
    print_values=True
    MAE_train = []
    MAE_test  = []
    R2_train  = []
    R2_test   = []

    n_train = my_xgb_X_train.shape[0]
    n_test  = my_xgb_X_test.shape[0]

    # fix a seed
    np.random.seed(1234)

    #repeat 'n_repeats' times
    for i in range(n_repeats):
        
        #select a subset
        idx_train = np.random.choice(n_train, int(subset_ratio*n_train))
        idx_test  = np.random.choice(n_test,  int(subset_ratio*n_test ))

        # Use a scaler for train and test partition
        X_train_ = scaler.fit_transform(my_xgb_X_train[idx_train])
        X_test_  = scaler.transform(my_xgb_X_test[idx_test])
        
        #generate prediction on the subset
        y_pred_train_ = my_xgb_tuned.predict(X_train_)
        y_pred_test_  =  my_xgb_tuned.predict(X_test_)

        #compute (and store) performances on the subset
        ## for MAE
        MAE_train.append(mean_absolute_error(my_xgb_y_train[idx_train], y_pred_train_))
        MAE_test.append(mean_absolute_error(my_xgb_y_test[idx_test], y_pred_test_))
        ## for R_squared
        R2_train.append(r2_score(my_xgb_y_train[idx_train], y_pred_train_))
        R2_test.append(r2_score(my_xgb_y_test[idx_test], y_pred_test_))    

    train_median_MAE, train_CI5_MAE, train_CI95_MAE = np.quantile(MAE_train, [0.5, 0.05, 0.95])
    test_median_MAE, test_CI5_MAE, test_CI95_MAE    = np.quantile(MAE_test, [0.5, 0.05, 0.95])
    train_median_r2, train_CI5_r2, train_CI95_r2 = np.quantile(R2_train, [0.5, 0.05, 0.95])
    test_median_r2, test_CI5_r2, test_CI95_r2    = np.quantile(R2_test, [0.5, 0.05, 0.95])    
    
    
    if print_values:
        print(f"MAE train (Random Forests): {train_median_MAE:.3f} [{train_CI5_MAE:.3f}-{train_CI95_MAE:.3f}]")
        print(f"MAE test (Random Forests): {test_median_MAE:.3f} [{test_CI5_MAE:.3f}-{test_CI95_MAE:.3f}]")
        print(f"R^2 train (Random Forests): {train_median_r2:.3f} [{train_CI5_r2:.3f}-{train_CI95_r2:.3f}]")
        print(f"R^2 test (Random Forests): {test_median_r2:.3f} [{test_CI5_r2:.3f}-{test_CI95_r2:.3f}]")

    my_ove = {
    'Alghorithm': ["XGBoost"] * 2,
    'Partition': ['Train', 'Test'],
    'MAE Median': [train_median_MAE,test_median_MAE],
    'MAE CI 5%': [train_CI5_MAE, test_CI5_MAE],
    'MAE CI 95%': [train_CI95_MAE, test_CI95_MAE],
    'R^2 Median': [train_median_r2, test_median_r2],
    'R^2 CI 5%': [train_CI5_r2, test_CI5_r2],
    'R^2 CI 95%': [train_CI95_r2, test_CI95_r2]
    }
    my_ove = pd.DataFrame(my_ove)
    my_ove.insert(loc=2, column='MAE', value=[mae_xgb_train,mae_xgb_test])
    my_ove.insert(loc=2, column='R^2', value=[r2_xgb_train,r2_xgb_test])
    print('-'*40)
    Data_3 = pd.DataFrame(
            {
                'Features': my_features,
                'Parameters': my_xgb_tuned.feature_importances_
            }
    ).round(3)
    Data_3= Data_3.sort_values(by='Parameters', ascending=False)
    Data_3






    
    coef = my_xgb_tuned.feature_importances_ 
    df_coef = pd.DataFrame({'feature': my_features, 'coef': coef})
    df_coef = df_coef.sort_values('coef', ascending=False)
    plt.figure()  # Create a new figure
    plt.bar(df_coef['feature'], df_coef['coef'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Coefficient")
    plt.tight_layout()
    plt.savefig(f'{my_dir}Feature_importante(XGBoost).png')
    plt.show()
    
    # SHAP summary plot
    explainer = shap.TreeExplainer(my_xgb_tuned)
    shap_values = explainer(my_xgb_X_test)
    
    plt.figure()  # Create a new figure
    shap.summary_plot(shap_values, 
                      feature_names=my_features,
                      show=False)
    
    # Save the SHAP plot
    plt.savefig(f'{my_dir}Shap_(XGBoost).png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return(my_ove)








    
    # ove_3 = overf_check(xgb_tuned,
    #                 title_overfit = 'Overfitting check - xgb',
    #                 my_dpi = 100,
    #                 print_values=True,
    #                 model_name= 'xgb')
    
    # ove_3.insert(loc=2, column='MAE', value=[mae_xgb_train,mae_xgb_test])
    # ove_3.insert(loc=2, column='R^2', value=[r2_xgb_train,r2_xgb_test])
    
    