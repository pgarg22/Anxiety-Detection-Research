#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:55:28 2022

@author: octopusphoenix
"""

from sklearn.model_selection import train_test_split
from functions_toolbox import  feature_selector_basic, recursive_feature_selector_cv,transform_categorical, scale_numerical_min_max,optimize_rf_hyperparameters, evaluate_model_performance

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.svm import SVC # "Support vector classifier"  

df_event_features= pd.read_csv("event_features_participants_all.csv")

X_forest=df_event_features.loc[:, ~df_event_features.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
y_forest=df_event_features['Condition']  #



X_forest=  feature_selector_basic(X_forest,y_forest,0.95)

best_features= recursive_feature_selector_cv(X_forest,y_forest,20)

X_forest=X_forest[best_features.tolist()]



y_forest=transform_categorical(y_forest)
X_forest=scale_numerical_min_max(X_forest)




X_train, X_test, y_train, y_test = train_test_split(X_forest, y_forest, test_size=0.3) # 70% training and 30% test





#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
evaluate_model_performance(clf,X_train, X_test, y_train, y_test, "Random Forest Base")



best_rf_hyperparam= optimize_rf_hyperparameters(X_train,y_train)


base_model = RandomForestClassifier(**best_rf_hyperparam)



evaluate_model_performance(base_model,X_train, X_test, y_train, y_test,"Random Forest Optimized")





# declare parameters
params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100
        }         
           



modelxgb = xgb.XGBClassifier(**params)
evaluate_model_performance(modelxgb,X_train, X_test, y_train, y_test ,"XGB")

    
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}


svm_linear = SVC(kernel='linear', random_state=0)  
svm_poly = SVC(kernel= 'poly', random_state=0)  
svm_rbf = SVC(kernel= 'rbf', random_state=0)  
evaluate_model_performance(svm_linear,X_train, X_test, y_train, y_test, "SVM Linear")
evaluate_model_performance(svm_poly,X_train, X_test, y_train, y_test, "SVM Poly")
evaluate_model_performance(svm_rbf,X_train, X_test, y_train, y_test, "SVM RBF")


