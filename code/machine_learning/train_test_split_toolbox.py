#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:54:43 2023

@author: octopusphoenix




"""


from ML_functions_toolbox import  feature_selector_basic,transform_categorical, scale_numerical_standard 
from machine_learning_toolbox import run_random_forest, run_xgb
from machine_learning_toolbox import run_svm
from ml_graph_toolbox import push_ml_results_norm, push_ml_results_trans

import pandas as pd

import os
from sklearn.model_selection import train_test_split




def participant_train_test_split(participant_id,df_event_features,test_name,directory):
    train= df_event_features[df_event_features['Participant'] != participant_id]
    test=  df_event_features[df_event_features['Participant'] == participant_id]
    
    X_train =train.loc[:, ~train.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    X_test =test.loc[:, ~test.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    y_train=train['Condition']  
    y_test=test['Condition']  



    X_train, train_scaler=scale_numerical_standard(X_train)
    X_test= scale_numerical_standard(X_test, train_scaler)
    y_train, encoder= transform_categorical(y_train)
    y_test=transform_categorical(y_test,encoder)

    X_train= feature_selector_basic(X_train,0.95)

    
    print("Paricipant :" + str(participant_id))
    
    rf_score= run_random_forest (X_train, X_test, y_train, y_test,test_name+"_Participant_"+str(participant_id),directory)
    xgb_score= run_xgb(X_train, X_test, y_train, y_test,test_name+"_Participant_"+str(participant_id),directory)
    svm_score= run_svm(X_train, X_test, y_train, y_test,test_name+"_Participant_"+str(participant_id),directory)


    return(rf_score+ xgb_score +svm_score)






def video_train_test_split(label_id,df_event_features,test_name,directory):
    train= df_event_features[df_event_features['Label'] != label_id]
    test=  df_event_features[df_event_features['Label'] == label_id]
    
    X_train =train.loc[:, ~train.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    X_test =test.loc[:, ~test.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    y_train=train['Condition']  
    y_test=test['Condition']  



    X_train, train_scaler=scale_numerical_standard(X_train)
    X_test= scale_numerical_standard(X_test, train_scaler)
    y_train, encoder= transform_categorical(y_train)
    y_test=transform_categorical(y_test,encoder)

    X_train= feature_selector_basic(X_train,0.95)

    
    print("Video :" + str(label_id))
    
    rf_score= run_random_forest (X_train, X_test, y_train, y_test,test_name+"_Video_"+str(label_id),directory)
    xgb_score= run_xgb(X_train, X_test, y_train, y_test,test_name+"_Video_"+str(label_id),directory)
    svm_score= run_svm(X_train, X_test, y_train, y_test,test_name+"_Video_"+str(label_id),directory)


    return(rf_score+ xgb_score +svm_score)





def all_video_transitional(df_event_features,test_name,directory):
    df_video = pd.DataFrame(columns=['rf_accuracy','rf_precision', 'rf_recall', 'rf_f1score',  
                                'xgb_accuracy', 'xgb_precision', 'xgb_recall', 'xgb_f1score',
                                'svc_accuracy','svc_precision', 'svc_recall', 'svc_f1score'])
    
    for i in range(1,9):
        
        rslt= video_train_test_split(i, df_event_features,test_name+"_Trans",directory)
        df_video.loc[len(df_video)] = rslt
        
    
    df_video.to_csv(directory+'Individual_Video_test_results_transitional.csv')
 
    
 
    
 
def all_participant_transitional(df_event_features,test_name,directory):

    df_participant = pd.DataFrame(columns=['rf_accuracy','rf_precision', 'rf_recall', 'rf_f1score',  
                                'xgb_accuracy', 'xgb_precision', 'xgb_recall', 'xgb_f1score',
                                'svc_accuracy','svc_precision', 'svc_recall', 'svc_f1score'])
 
    for i in range(1,20):
        
        rslt= participant_train_test_split(i, df_event_features,test_name+"_Trans",directory)
        df_participant.loc[len(df_participant)] = rslt
        
    df_participant.to_csv(directory+'Individual_Participant_test_results_transisitional.csv')




def all_participant_test(df_event_features,test_name,directory):


    df_participant = pd.DataFrame(columns=['rf_accuracy','rf_precision', 'rf_recall', 'rf_f1score',  
                                'xgb_accuracy', 'xgb_precision', 'xgb_recall', 'xgb_f1score',
                                'svc_accuracy','svc_precision', 'svc_recall', 'svc_f1score'])
    
    for i in range(1,20):
        
        rslt= participant_train_test_split(i, df_event_features,test_name+"_Norm",directory)
        df_participant.loc[len(df_participant)] = rslt
        
    df_participant.to_csv(directory+'Individual_Participant_test_results.csv')




def all_video_test(df_event_features,test_name,directory):
    
    df_video = pd.DataFrame(columns=['rf_accuracy','rf_precision', 'rf_recall', 'rf_f1score',  
                                'xgb_accuracy', 'xgb_precision', 'xgb_recall', 'xgb_f1score',
                                'svc_accuracy','svc_precision', 'svc_recall', 'svc_f1score'])
    
        
    for i in range(1,9):
        
        rslt= video_train_test_split(i, df_event_features,test_name+"_Norm",directory)
        df_video.loc[len(df_video)] = rslt
        
    
    df_video.to_csv(directory+'Individual_Video_test_results.csv')



def random_split_test(df_event_features,test_name,directory):

        
    X=df_event_features.loc[:, ~df_event_features.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    y=df_event_features['Condition']  #

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,stratify=y, random_state=42) # 70% training and 30% test

    X_train, train_scaler=scale_numerical_standard(X_train)
    X_test= scale_numerical_standard(X_test, train_scaler)
    y_train, encoder= transform_categorical(y_train)
    y_test=transform_categorical(y_test,encoder)

    run_random_forest (X_train, X_test, y_train, y_test,test_name+"_Random_split",directory)
    run_xgb(X_train, X_test, y_train, y_test,test_name+"_Random_split",directory)
    run_svm(X_train, X_test, y_train, y_test,test_name+"_Random_split",directory)
    
    
    
def run_all_test_norm(df,test_name,directory):
    
    
    
    random_split_test(df,test_name+"_Norm",directory)
    all_participant_test(df, test_name,directory)
    all_video_test(df, test_name,directory)
    push_ml_results_norm(directory)

    
def run_all_test_trans(df_trans,test_name,directory):
    
    
    random_split_test(df_trans,test_name+"_Trans",directory)
    all_participant_transitional(df_trans, test_name,directory)
    all_video_transitional(df_trans, test_name,directory)
    push_ml_results_trans(directory)
        
    
def run_multiple_tests_norm(df,test_name_list):
    
    if not os.path.exists("test_results"):
        os.mkdir("test_results")

    res_directory = os.getcwd() + "/test_results/"
    for i in range(0,len(test_name_list)):
        
        test_name= test_name_list[i]
        
        if not os.path.exists(res_directory+test_name):
            os.mkdir(res_directory+test_name)
        
        directory = res_directory+test_name+"/" 
        run_all_test_norm(df,test_name,directory)
    
    
    
    
    
def run_multiple_tests_trans(df,test_name_list):
    
    if not os.path.exists("test_results"):
        os.mkdir("test_results")

    res_directory = os.getcwd() + "/test_results/"
    for i in range(0,len(test_name_list)):
        
        test_name= test_name_list[i]
        
        if not os.path.exists(res_directory+test_name):
            os.mkdir(res_directory+test_name)
        
        directory = res_directory+test_name+"/" 
        run_all_test_trans(df,test_name,directory)
    
    
    
   
