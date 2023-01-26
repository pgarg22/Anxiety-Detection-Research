#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:54:37 2023

@author: octopusphoenix
"""

from sklearn.ensemble import RandomForestClassifier
from functions_toolbox import  recursive_feature_selector_cv,optimize_rf_hyperparameters, evaluate_model_performance
import xgboost as xgb
from sklearn.svm import SVC # "Support vector classifier"  



"""
============================================================================================================
Function to run random forest ML algorithm along with feature selection
============================================================================================================
"""

def run_random_forest(X_train, X_test, y_train, y_test, n_estimators):
 
     clf=RandomForestClassifier(n_estimators=n_estimators)
     best_features= recursive_feature_selector_cv(X_train,y_train,clf,20,'accuracy')
     X_train=X_train[best_features.tolist()]
     X_test=X_test[best_features.tolist()]
    
     
     best_rf_hyperparam= optimize_rf_hyperparameters(X_train,y_train)
     
     optimized_model = RandomForestClassifier(**best_rf_hyperparam)
     
     return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test,"Random Forest Optimized"))
     



"""
============================================================================================================
Function to run XGB ML algorithm along with feature selection
============================================================================================================
"""



def run_xgb(X_train, X_test, y_train, y_test):
     
     # declare parameters
     params = {
                 'objective':'binary:logistic',
                 'max_depth': 4,
                 'alpha': 10,
                 'learning_rate': 1.0,
                 'n_estimators':100
             }         
     modelxgb = xgb.XGBClassifier(**params)
     best_features= recursive_feature_selector_cv(X_train,y_train,modelxgb,20,'accuracy')
     X_train=X_train[best_features.tolist()]
     X_test=X_test[best_features.tolist()]
     
     modelxgb = xgb.XGBClassifier(**params)
     
     return(evaluate_model_performance(modelxgb,X_train, X_test, y_train, y_test ,"XGB"))           


"""
============================================================================================================
Function to run SVC ML algorithm along with feature selection
============================================================================================================
"""


def run_svc_linear(X_train, X_test, y_train, y_test):
     
    svm_linear = SVC(kernel='linear', random_state=0)  

    best_features= recursive_feature_selector_cv(X_train,y_train,svm_linear,20,'accuracy')
    X_train=X_train[best_features.tolist()]
    X_test=X_test[best_features.tolist()]
     
    svm_linear = SVC(kernel='linear', random_state=0)  
    return(evaluate_model_performance(svm_linear,X_train, X_test, y_train, y_test, "SVM Linear"))




"""
============================================================================================================
Function to run random forest ML algorithm along without feature selection
============================================================================================================
"""

def run_random_forest_without_fs(X_train, X_test, y_train, y_test, n_estimators):
 
     clf=RandomForestClassifier(n_estimators=n_estimators)
     
     best_rf_hyperparam= optimize_rf_hyperparameters(X_train,y_train)
     
     optimized_model = RandomForestClassifier(**best_rf_hyperparam)
     
     return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test,"Random Forest Optimized"))
     



"""
============================================================================================================
Function to run XGB ML algorithm along without feature selection
============================================================================================================
"""



def run_xgb_without_fs(X_train, X_test, y_train, y_test):
     
     # declare parameters
     params = {
                 'objective':'binary:logistic',
                 'max_depth': 4,
                 'alpha': 10,
                 'learning_rate': 1.0,
                 'n_estimators':100
             }         
     modelxgb = xgb.XGBClassifier(**params)
   
     
     return(evaluate_model_performance(modelxgb,X_train, X_test, y_train, y_test ,"XGB"))           


"""
============================================================================================================
Function to run SVC ML algorithm along without feature selection
============================================================================================================
"""


def run_svc_linear_without_fs(X_train, X_test, y_train, y_test):
     
 
     
    svm_linear = SVC(kernel='linear', random_state=0)  
    return(evaluate_model_performance(svm_linear,X_train, X_test, y_train, y_test, "SVM Linear"))

     