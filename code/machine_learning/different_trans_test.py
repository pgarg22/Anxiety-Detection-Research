#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:17:55 2023

@author: octopusphoenix
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:01:19 2023

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

names=["10","20","30","10_onlybegin","20_onlybegin",
       "30_onlybegin","40_onlybegin"]


for c in  names:
    df_event_features_trans= pd.read_csv("merged_processed_files/ECG_RSP_Featurestrans"+c+".csv")
    X_trans=df_event_features_trans.loc[:, ~df_event_features_trans.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    y_trans=df_event_features_trans['Condition']  #
    X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_trans, y_trans, test_size=0.2, stratify=y_trans, random_state=42)
    
    y_train_trans, encoder_trans= transform_categorical(y_train_trans)
    X_train_trans, train_scaler_trans=scale_numerical_standard(X_train_trans)
    X_train_trans= feature_selector_basic(X_train_trans,0.9)
    
    df_feature_selector_trans, features_list_trans, selector_names_trans= run_all_selectors(X_train_trans,y_train_trans,num_feats)
    df_feature_selector_trans.to_csv("features_list_transitional"+ c+".csv")
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
        
    for n,name in enumerate(selector_names_trans):
        selector_names_trans[n]=selector_names_trans[n]+"_"+c
    run_multiple_tests_without_fs_trans(df_event_features_trans,
                                  features_list_trans,selector_names_trans)