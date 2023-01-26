#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 17:17:10 2023

@author: octopusphoenix
"""


from train_test_split_toolbox_without_fs  import run_multiple_tests_without_fs_norm
from train_test_split_toolbox_without_fs  import run_multiple_tests_without_fs_trans
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



df_event_features= pd.read_csv("event_features_ecg_rsp.csv")
num_feats=10


features_list=[]
selector_names=[]

features_list.append(['HRV_MinNN', 'RRV_LFn', 'RRV_SD2'])
selector_names.append("Exhaustive Feature selector")

for feat in features_list:
    feat.append('Condition')
    feat.append('Label')
    feat.append('Unnamed: 0')
    feat.append('Participant')
run_multiple_tests_without_fs_norm(df_event_features,
                              features_list,selector_names)



df_event_features_trans= pd.read_csv("event_features_ecg_rsp_transitional_phase.csv")


features_list_trans=[]
selector_names_trans=[]
features_list_trans.append(['HRV_MinNN', 'RRV_LFn', 'RRV_SD2'])
selector_names_trans.append("Exhaustive Feature selector")
for feat in features_list_trans:
    feat.append('Condition')
    feat.append('Label')
    feat.append('Unnamed: 0')
    feat.append('Participant')
run_multiple_tests_without_fs_trans(df_event_features_trans,
                              features_list_trans,selector_names_trans)