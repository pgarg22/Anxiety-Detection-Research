#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:22:51 2022

@author: octopusphoenix
"""

from Load_ecg_res_data import *
from functions_toolbox import clean_missing_values
import numpy as np
import neurokit2 as nk
import pandas as pd



#############################################################################################
#Setting up event info


event_arr= np.arange(0, 1230000, 30000, dtype=int)
interval_related_features_ecg=[]


#############################################################################################




for k in range(0,len(data_ecg_res)):
    
    # Process ecg
    if(k==12):
        continue;
    data_clean, info= nk.bio_process(ecg=data_ecg_res[k][:,0], sampling_rate=500)
    # ecg_signals, info = nk.ecg_process(data_ecg_res[k][:,0], sampling_rate=500)
    epochs = nk.epochs_create(data_clean, 
                              events=event_arr, 
                              sampling_rate=500, 
                              epochs_start=0, 
                              epochs_end=60)
    
    df2=nk.ecg_intervalrelated(epochs)
    interval_related_features_ecg.append(df2)
    
    
    
df_interval_features = pd.concat(interval_related_features_ecg)  

nan_cols= df_interval_features.columns[df_interval_features.isnull().any()]

df_interval_features= clean_missing_values(df_event_features,50, median_cols=nan_cols)
    
    