#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 11:47:30 2023

@author: octopusphoenix
"""

import pandas as pd
from train_test_split_toolbox_without_fs  import run_multiple_tests_without_fs_norm

df_event_featurestest= pd.read_csv("event_features_ecg_rsp.csv")
df_test= pd.read_csv("features_list_normal.csv")
features_listtest=[]
selector_namestest=[]
# best_featurestest= df_test.head(10)["Feature"].to_list()                                  
# features_listtest.append(best_featurestest)

test_features= df_test['Feature'][df_test['Mutual Info Selector'] == True].tolist()
                           
features_listtest.append(test_features)
selector_namestest.append("Mutual Info Selector")


# selector_namestest.append("Best All selectors")
# features_listtest.append(['HRV_MinNN', 'RRV_LFn', 'RRV_SD2'])
# selector_namestest.append("Exhaustive Feature selector")

for feat in features_listtest:
    feat.append('Condition')
    feat.append('Label')
    feat.append('Unnamed: 0')
    feat.append('Participant')
run_multiple_tests_without_fs_norm(df_event_featurestest,
                              features_listtest,selector_namestest)