#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:16:28 2023

@author: octopusphoenix
"""


from train_test_split_toolbox import participant_train_test_split

import pandas as pd
import warnings
warnings.filterwarnings("ignore")




df_event_features= pd.read_csv("event_features_ecg_rsp_transitional_phase.csv")

df_participant = pd.DataFrame(columns=['rf_precision', 'rf_recall', 'rf_f1score', 'rf_accuracy', 
                           'xgb_precision', 'xgb_recall', 'xgb_f1score', 'xgb_accuracy', 
                           'svc_precision', 'svc_recall', 'svc_f1score', 'svc_accuracy'])

for i in range(1,20):
    
    rslt= participant_train_test_split(i, df_event_features)
    df_participant.loc[len(df_participant)] = rslt
    
    
df_participant.to_csv('Individual_Participant_test_results_transisitional.csv')

