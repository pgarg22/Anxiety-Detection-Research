#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:54:37 2023

@author: octopusphoenix
"""

from sklearn.ensemble import RandomForestClassifier
from ML_functions_toolbox import  recursive_feature_selector_cv,optimize_rf_hyperparameters, evaluate_model_performance
from ML_functions_toolbox import optimize_svc_hyperparameters, optimize_xgb_hyperparameters
import xgboost as xgb
from sklearn.svm import SVC # "Support vector classifier"  



"""
============================================================================================================
Function to run random forest ML algorithm along with feature selection
============================================================================================================
"""

def run_random_forest(X_train, X_test, y_train, y_test,test_name,directory):
 
     clf=RandomForestClassifier(random_state=0)
     best_features= recursive_feature_selector_cv(X_train,y_train,clf,20,'accuracy')
     X_train=X_train[best_features.tolist()]
     X_test=X_test[best_features.tolist()]
    
     
     best_rf_hyperparam= optimize_rf_hyperparameters(X_train,y_train)
     
     optimized_model = RandomForestClassifier(**best_rf_hyperparam)
     
     return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test,test_name+"_Random Forest",directory))
     

"""
============================================================================================================
Function to run XGB ML algorithm along with feature selection
============================================================================================================
"""

def run_xgb(X_train, X_test, y_train, y_test,test_name,directory): 
     # declare parameters
     params = {
                 'objective':'binary:logistic',
                 'max_depth': 4,
                 'alpha': 10,
                 'learning_rate': 0.1,
                 'n_estimators':100
             }         
     modelxgb = xgb.XGBClassifier(**params)
     best_features= recursive_feature_selector_cv(X_train,y_train,modelxgb,20,'accuracy')
     X_train=X_train[best_features.tolist()]
     X_test=X_test[best_features.tolist()]
     
     
     best_xgb_hyperparam= optimize_xgb_hyperparameters(X_train,y_train)
     optimized_model = xgb.XGBClassifier(**best_xgb_hyperparam)   
     
     return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test ,test_name+"_XGB",directory))           


"""
============================================================================================================
Function to run SVM ML algorithm along with feature selection
============================================================================================================
"""


def run_svm(X_train, X_test, y_train, y_test,test_name,directory):
     
    svm_linear = SVC(random_state=0)  
    best_features= recursive_feature_selector_cv(X_train,y_train,svm_linear,20,'accuracy')
    X_train=X_train[best_features.tolist()]
    X_test=X_test[best_features.tolist()]
     
    
    best_svc_hyperparam= optimize_svc_hyperparameters(X_train,y_train)
    optimized_model = SVC(**best_svc_hyperparam)
    return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test, test_name+"_SVM Linear",directory))


"""
============================================================================================================
Function to run random forest ML algorithm along without feature selection
============================================================================================================
"""

def run_random_forest_without_fs(X_train, X_test, y_train, y_test,test_name,directory):
 
     best_rf_hyperparam= optimize_rf_hyperparameters(X_train,y_train)
     optimized_model = RandomForestClassifier(**best_rf_hyperparam)
     return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test,test_name+"_Random Forest",directory))
     

"""
============================================================================================================
Function to run XGB ML algorithm along without feature selection
============================================================================================================
"""

def run_xgb_without_fs(X_train, X_test, y_train, y_test,test_name,directory):
     
     best_xgb_hyperparam= optimize_xgb_hyperparameters(X_train,y_train)
     optimized_model = xgb.XGBClassifier(**best_xgb_hyperparam)       
     return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test ,test_name+"_XGB",directory))           


"""
============================================================================================================
Function to run SVM ML algorithm along without feature selection
============================================================================================================
"""

def run_svm_without_fs(X_train, X_test, y_train, y_test,test_name,directory):
     
    best_svc_hyperparam= optimize_svc_hyperparameters(X_train,y_train)
    optimized_model = SVC(**best_svc_hyperparam)
    return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test, test_name+ "_SVM",directory))

     