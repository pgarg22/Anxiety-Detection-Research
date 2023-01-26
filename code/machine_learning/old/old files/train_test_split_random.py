#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:04:40 2023

@author: octopusphoenix
"""

from sklearn.model_selection import train_test_split
from ML_functions_toolbox import  feature_selector_basic,transform_categorical, scale_numerical_min_max 
from machine_learning_toolbox import run_random_forest, run_xgb
from machine_learning_toolbox import run_svc_linear
import numpy as np
import pandas as pd


df_event_features= pd.read_csv("event_features_ecg_rsp.csv")

X=df_event_features.loc[:, ~df_event_features.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
y=df_event_features['Condition']  #

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

X_train, train_scaler=scale_numerical_min_max(X_train)
X_test= scale_numerical_min_max(X_test, train_scaler)
y_train, encoder= transform_categorical(y_train)
y_test=transform_categorical(y_test,encoder)

run_random_forest (X_train, X_test, y_train, y_test,100)
run_xgb(X_train, X_test, y_train, y_test)
run_svc_linear(X_train, X_test, y_train, y_test)