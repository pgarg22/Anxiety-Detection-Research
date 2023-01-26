#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:34:16 2023

@author: octopusphoenix
"""



"""
============================================================================================================
Importing modules
============================================================================================================
"""

import pandas as pd
from ml_graph_toolbox import  push_viz_scatter_subplots2



"""
============================================================================================================
Loading data
============================================================================================================
"""


df_metadata= pd.read_csv("Metadata.csv")
cols=["Participant ID","Beck Anxiety","Hamilton Anxiety"]
df_metadata=df_metadata[cols]


df_participant_results= pd.read_csv("Individual_Participant_test_results.csv")
df_participant_results.rename(columns = {'Unnamed: 0':'Participant ID'}, inplace = True)
df_participant_results['Participant ID']=df_participant_results['Participant ID'] +1
new_df = pd.merge(df_participant_results, df_metadata,  how='left', left_on=['Participant ID'], right_on = ['Participant ID'])


"""
============================================================================================================
Graph 1
============================================================================================================
"""


colors= ["green", "orange", "red","blue"]
graph_title= 'Individual Participant test results'
subtitles= ['rf_precision', 'rf_recall', 'rf_f1score', 'rf_accuracy', 
            'xgb_precision', 'xgb_recall', 'xgb_f1score', 'xgb_accuracy', 
            'svc_precision', 'svc_recall', 'svc_f1score', 'svc_accuracy',
            "Beck Anxiety","Hamilton Anxiety"]

ytitle="Values"
mode= 'markers+lines'
symbols= ["circle","diamond","cross","arrow"]
push_viz_scatter_subplots2(4,4,new_df,graph_title,subtitles ,ytitle,mode ,symbols,colors)



"""
============================================================================================================
Graph2
============================================================================================================
"""



df_video_results= pd.read_csv("Individual_Video_test_results.csv")
df_video_results.rename(columns = {'Unnamed: 0':'Video'}, inplace = True)
df_video_results['Video']=df_video_results['Video'] +1
cols=["Video","rf_accuracy","xgb_accuracy","svc_accuracy"]
df_video_results=df_video_results[cols]


graph_title= 'Individual Video test results'
subtitles= ['rf_accuracy','xgb_accuracy', 'svc_accuracy']

push_viz_scatter_subplots2(1,3,df_video_results,graph_title,subtitles ,ytitle,mode ,symbols,colors)




"""
============================================================================================================
Graph3
============================================================================================================
"""


df_participant_results_trans= pd.read_csv("Individual_Participant_test_results_transisitional.csv")
df_participant_results_trans.rename(columns = {'Unnamed: 0':'Participant ID'}, inplace = True)
df_participant_results_trans['Participant ID']=df_participant_results_trans['Participant ID'] +1

new_df_trans = pd.merge(df_participant_results_trans, df_metadata,  how='left', left_on=['Participant ID'], right_on = ['Participant ID'])



graph_title= 'Individual Participant test results with transitional period'
subtitles= ['rf_precision', 'rf_recall', 'rf_f1score', 'rf_accuracy', 
            'xgb_precision', 'xgb_recall', 'xgb_f1score', 'xgb_accuracy', 
            'svc_precision', 'svc_recall', 'svc_f1score', 'svc_accuracy',
            "Beck Anxiety","Hamilton Anxiety"]


push_viz_scatter_subplots2(4,4,new_df_trans,graph_title,subtitles ,ytitle,mode ,symbols,colors)



"""
============================================================================================================
Graph4
============================================================================================================
"""

graph_title= 'Individual Video test results with transitional period'


subtitles= ['rf_accuracy','xgb_accuracy', 'svc_accuracy']

df_video_results_trans= pd.read_csv("Individual_Video_test_results_transitional.csv")

df_video_results_trans.rename(columns = {'Unnamed: 0':'Video'}, inplace = True)
df_video_results_trans['Video']=df_video_results_trans['Video'] +1
cols=["Video","rf_accuracy","xgb_accuracy","svc_accuracy"]

df_video_results_trans=df_video_results_trans[cols]

fig= push_viz_scatter_subplots2(1,3,df_video_results_trans,graph_title,subtitles ,ytitle,mode ,symbols,colors)




