#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:37:44 2023

@author: octopusphoenix
"""


from train_test_split_toolbox_without_fs  import run_multiple_tests_without_fs_norm
from train_test_split_toolbox_without_fs  import run_multiple_tests_without_fs_trans
import pandas as pd
from feature_selection_toolbox import run_all_selectors
from sklearn.model_selection import train_test_split
from ML_functions_toolbox import feature_selector_basic, transform_categorical, scale_numerical_standard
import warnings
warnings.filterwarnings("ignore")


num_feats=10


df_event_features= pd.read_csv("event_features_ecg_rsp.csv")





df_metadata= pd.read_csv("Metadata.csv")
cols=["Participant ID","Beck Anxiety","Hamilton Anxiety"]
df_metadata=df_metadata[cols]



df_event_features_BH = pd.merge(df_event_features, df_metadata,  how='left', left_on=['Participant'], right_on = ['Participant ID'])



X=df_event_features.loc[:, ~df_event_features.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
y=df_event_features['Condition']  #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

y_train, encoder= transform_categorical(y_train)
X_train, train_scaler=scale_numerical_standard(X_train)
X_train= feature_selector_basic(X_train,0.9)

df_feature_selector, features_list, selector_names= run_all_selectors(X_train,y_train,num_feats)
df_feature_selector.to_csv("features_list_normal_BH.csv")
best_features= df_feature_selector.head(10)["Feature"].to_list()                                  
features_list.append(best_features)

selector_names.append("Best All selectors")
# features_list.append(['HRV_MinNN', 'RRV_LFn', 'RRV_SD2'])
# selector_names.append("Exhaustive Feature selector")

for feat in features_list:
    feat.append('Condition')
    feat.append('Label')
    feat.append('Unnamed: 0')
    feat.append('Participant')
    feat.append('Beck Anxiety')
    feat.append('Hamilton Anxiety')
run_multiple_tests_without_fs_norm(df_event_features_BH,
                              features_list,selector_names)








df_event_features_trans= pd.read_csv("event_features_ecg_rsp_transitional_phase.csv")

df_event_features_trans_BH = pd.merge(df_event_features_trans, df_metadata,  how='left', left_on=['Participant'], right_on = ['Participant ID'])




X_trans=df_event_features_trans.loc[:, ~df_event_features_trans.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
y_trans=df_event_features_trans['Condition']  #
X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_trans, y_trans, test_size=0.2, stratify=y_trans, random_state=42)

y_train_trans, encoder_trans= transform_categorical(y_train_trans)
X_train_trans, train_scaler_trans=scale_numerical_standard(X_train_trans)
X_train_trans= feature_selector_basic(X_train_trans,0.9)

df_feature_selector_trans, features_list_trans, selector_names_trans= run_all_selectors(X_train_trans,y_train_trans,num_feats)
df_feature_selector_trans.to_csv("features_list_transitional.csv")
best_features_trans= df_feature_selector_trans.head(10)["Feature"].to_list()                                  
features_list_trans.append(best_features_trans)
selector_names_trans.append("Best All selectors")
# features_list_trans.append(['HRV_MinNN', 'RRV_LFn', 'RRV_SD2'])
# selector_names_trans.append("Exhaustive Feature selector")
for feat in features_list_trans:
    feat.append('Condition')
    feat.append('Label')
    feat.append('Unnamed: 0')
    feat.append('Participant')
    feat.append('Beck Anxiety')
    feat.append('Hamilton Anxiety')
run_multiple_tests_without_fs_trans(df_event_features_trans_BH,
                              features_list_trans,selector_names_trans)
