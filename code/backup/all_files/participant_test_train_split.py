#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:09:35 2023

@author: octopusphoenix
"""


from sklearn.model_selection import train_test_split
from functions_toolbox import  feature_selector_basic,transform_categorical, scale_numerical_min_max 
from machine_learning_toolbox import run_random_forest, run_xgb
from machine_learning_toolbox import run_svc_linear
import numpy as np
import pandas as pd

import random

import warnings
warnings.filterwarnings("ignore")




df_event_features= pd.read_csv("event_features_ecg_rsp.csv")


participant_id= random.randint(1,19)


train= df_event_features[df_event_features['Participant'] != participant_id]
test=  df_event_features[df_event_features['Participant'] == participant_id]

                         

X_train =train.loc[:, ~train.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
X_test =test.loc[:, ~test.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]



y_train=train['Condition']  
y_test=test['Condition']  



X_train, train_scaler=scale_numerical_min_max(X_train)
X_test= scale_numerical_min_max(X_test, train_scaler)

y_train, encoder= transform_categorical(y_train)
y_test=transform_categorical(y_test,encoder)

X_train= feature_selector_basic(X_train,0.95)




run_random_forest (X_train, X_test, y_train, y_test,100)


run_xgb(X_train, X_test, y_train, y_test)



run_svc_linear(X_train, X_test, y_train, y_test)

