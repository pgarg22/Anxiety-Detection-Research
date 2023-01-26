#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:54:45 2022

This is a script file containing all the functions for the anxiety  research

@author: octopusphoenix
"""


"""
============================================================================================================
Importing modules
============================================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV



from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from graphs_toolbox import push_heatmap
import os

"""
============================================================================================================
Function to calculate BMI
============================================================================================================
"""

def get_bmi(height,weight):
    return np.round(((weight*4535.9237)/(height*height))) #height in cm and weight in lbs



"""
============================================================================================================
Function to see machine learning model performance
============================================================================================================
"""

def evaluate_model_performance(model,X_train,X_test,y_train,y_test,name=""):
    model.fit(X_train, y_train)
    labels = [0, 1]
    y_pred = model.predict(X_test)
    cm= confusion_matrix(y_test,y_pred)
    push_heatmap(cm,name)
    # labels = ["True Neg","False Pos","False Neg","True Pos"]
    plot_confusion_matrix(model, X_test, y_test, cmap='GnBu')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix for model - '+ name)
    fig.colorbar(cax)
    if not os.path.exists("confusion_matrix"):
        os.mkdir("confusion_matrix")
    plt.savefig("confusion_matrix/"+str(name)+".png", facecolor='w', bbox_inches="tight",
            pad_inches=0.3, transparent=True)
    
    precision=precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1score= f1_score(y_test, y_pred)
    accuracy= accuracy_score(y_test, y_pred)
    
    print("**********************************************************************************")
    print("Model : " + name)
    print('Precision: %.3f' % precision )
    print('Recall: %.3f' % recall)
    print('F1: %.3f' % f1score)
    print('Accuracy: %.3f' % accuracy)
    print("**********************************************************************************")
    
    return([precision, recall, f1score, accuracy])

    


"""
============================================================================================================
Function to clean missing values and nan's'
============================================================================================================
"""    
   
    
def clean_missing_values(df,miss_thres_perc, mean_cols=[], median_cols=[],mode_cols=[]):
    
    min_count =  int(((100-miss_thres_perc)/100)*df.shape[0] + 1)
    df= df.dropna(axis=1, 
                thresh=min_count)
    for value in mean_cols:
        if value in df.columns:
            df[value]= df[value].fillna(df[value].mean()) 	
    
    for value in median_cols:
        if value in df.columns:
            df[value]= df[value].fillna(df[value].median()) 	

    for value in mode_cols:
        if value in df.columns:
            df[value]= df[value].fillna(df[value].mode()[0]) 	

    return(df)
    


"""
============================================================================================================
Function to do some basic filtering of features
============================================================================================================
"""
    
def feature_selector_basic(X_train,var_threshold):
    
    
    sel = VarianceThreshold(threshold=0.05)  # 0.01 indicates 99% of observations approximately
    sel.fit(X_train)  
    cols= X_train.columns[sel.get_support()]
    X_new= X_train[cols]
    
    cor_matrix = X_new.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > var_threshold)]
    filtered_X = X_train.drop(to_drop, axis=1)
    return(filtered_X)




"""
============================================================================================================
Function to select best features using recursive feature selection with cross validation
============================================================================================================
"""



def recursive_feature_selector_cv(X_train,y_train,classifier,crossvaltimes,scoring):
    
    rfecv = RFECV(estimator=classifier, step=1, cv=crossvaltimes,scoring=scoring)  
    rfecv = rfecv.fit(X_train, y_train)
    return( X_train.columns[rfecv.support_])

    


"""
============================================================================================================
Function to transform categorical columns into integer using label encoder
============================================================================================================
"""

def transform_categorical(df,label_encoder=None):
    flag=0
    if label_encoder==None:
        label_encoder = LabelEncoder()
        flag=1
    
    df= label_encoder.fit_transform(df)
    if flag==1:
        return(df,label_encoder)
    return (df)
        
    
    



"""
============================================================================================================
Function to scale numerical columns into integer using MinMax Scaler 

Returns scaler if none provided while calling the function
============================================================================================================
"""
def scale_numerical_min_max(data,scaler=None):
    flag=0
    if scaler==None :
        scaler = MinMaxScaler()
        flag=1
    data[data.columns] = scaler.fit_transform(data[data.columns])
    if flag==1:
        return(data,scaler)
    return (data)
    



"""
============================================================================================================
Function to scale numerical columns into integer using Standard Scaler 
============================================================================================================
"""
def scale_numerical_standard(data):
    scaler = StandardScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])
    return(data)
    
    



"""
============================================================================================================
Function to optimize random forest hyperparameters
============================================================================================================
"""
def optimize_rf_hyperparameters(X_train,y_train):
    
# Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)    
    return(rf_random.best_params_)
    




    