#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:54:43 2023

@author: octopusphoenix




"""


from functions_toolbox import  feature_selector_basic,transform_categorical, scale_numerical_min_max 
from machine_learning_toolbox import run_random_forest, run_xgb
from machine_learning_toolbox import run_svc_linear




def participant_train_test_split(participant_id,df_event_features):
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

    
    print("Paricipant :" + str(participant_id))
    
    rf_score= run_random_forest (X_train, X_test, y_train, y_test,100)
    xgb_score= run_xgb(X_train, X_test, y_train, y_test)
    svc_score= run_svc_linear(X_train, X_test, y_train, y_test)


    return(rf_score+ xgb_score +svc_score)






def video_train_test_split(label_id,df_event_features):
    train= df_event_features[df_event_features['Label'] != label_id]
    test=  df_event_features[df_event_features['Label'] == label_id]
    
    X_train =train.loc[:, ~train.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    X_test =test.loc[:, ~test.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    y_train=train['Condition']  
    y_test=test['Condition']  



    X_train, train_scaler=scale_numerical_min_max(X_train)
    X_test= scale_numerical_min_max(X_test, train_scaler)
    y_train, encoder= transform_categorical(y_train)
    y_test=transform_categorical(y_test,encoder)

    X_train= feature_selector_basic(X_train,0.95)

    
    print("Video :" + str(label_id))
    
    rf_score= run_random_forest (X_train, X_test, y_train, y_test,100)
    xgb_score= run_xgb(X_train, X_test, y_train, y_test)
    svc_score= run_svc_linear(X_train, X_test, y_train, y_test)


    return(rf_score+ xgb_score +svc_score)







