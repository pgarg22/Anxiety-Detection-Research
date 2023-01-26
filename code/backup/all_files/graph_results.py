#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:34:16 2023

@author: octopusphoenix
"""

import pandas as pd
from graphs_toolbox import  push_viz_scatter2,push_viz_scatter_subplots2

df_participant_results= pd.read_csv("Individual_Participant_test_results.csv")

# df_participant_results['Participant'] = df_participant_results['Unnamed: 0'].apply(lambda x: x+1)
df_participant_results.rename(columns = {'Unnamed: 0':'Participant'}, inplace = True)
df_participant_results['Participant']=df_participant_results['Participant'] +1
# cols=["Participant","rf_accuracy","xgb_precision","svc_precision"]

# df_participant_results=df_participant_results[cols]

push_viz_scatter_subplots2(4,4,df_participant_results,'Individual Participant test results', ['rf_precision', 'rf_recall', 'rf_f1score', 'rf_accuracy', 
                           'xgb_precision', 'xgb_recall', 'xgb_f1score', 'xgb_accuracy', 
                           'svc_precision', 'svc_recall', 'svc_f1score', 'svc_accuracy'],"Percentage",'markers+lines',["diamond","cross","arrow","diamond","cross","arrow","diamond","cross","arrow"])





df_video_results= pd.read_csv("Individual_Video_test_results.csv")

df_video_results.rename(columns = {'Unnamed: 0':'Video'}, inplace = True)
df_video_results['Video']=df_video_results['Video'] +1
cols=["Video","rf_accuracy","xgb_accuracy","svc_accuracy"]

df_video_results=df_video_results[cols]

push_viz_scatter_subplots2(3,1,df_video_results,'Individual Video test results', ['rf_accuracy', 
                            'xgb_accuracy', 
                           'svc_accuracy'],"Percentage",'markers+lines',["diamond","cross","arrow","diamond","cross","arrow","diamond","cross","arrow"])






df_participant_results_trans= pd.read_csv("Individual_Participant_test_results_transisitional.csv")

# df_participant_results['Participant'] = df_participant_results['Unnamed: 0'].apply(lambda x: x+1)
df_participant_results_trans.rename(columns = {'Unnamed: 0':'Participant'}, inplace = True)
df_participant_results_trans['Participant']=df_participant_results_trans['Participant'] +1
# cols=["Participant","rf_accuracy","xgb_precision","svc_precision"]

# df_participant_results=df_participant_results[cols]

push_viz_scatter_subplots2(4,4,df_participant_results_trans,'Individual Participant test results with transitional period', ['rf_precision', 'rf_recall', 'rf_f1score', 'rf_accuracy', 
                           'xgb_precision', 'xgb_recall', 'xgb_f1score', 'xgb_accuracy', 
                           'svc_precision', 'svc_recall', 'svc_f1score', 'svc_accuracy'],"Percentage",'markers+lines',["diamond","cross","arrow","diamond","cross","arrow","diamond","cross","arrow"])





df_video_results_trans= pd.read_csv("Individual_Video_test_results_transitional.csv")

df_video_results_trans.rename(columns = {'Unnamed: 0':'Video'}, inplace = True)
df_video_results_trans['Video']=df_video_results_trans['Video'] +1
cols=["Video","rf_accuracy","xgb_accuracy","svc_accuracy"]

df_video_results_trans=df_video_results_trans[cols]

push_viz_scatter_subplots2(3,1,df_video_results_trans,'Individual Video test results with transitional period', ['rf_accuracy', 
                            'xgb_accuracy', 
                           'svc_accuracy'],"Percentage",'markers+lines',["diamond","cross","arrow","diamond","cross","arrow","diamond","cross","arrow"])




