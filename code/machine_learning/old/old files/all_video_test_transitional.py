#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:17:43 2023

@author: octopusphoenix
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:48:51 2023

@author: octopusphoenix
"""



from train_test_split_toolbox import video_train_test_split

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


df_event_features= pd.read_csv("event_features_ecg_rsp_transitional_phase.csv")

df_video = pd.DataFrame(columns=['rf_precision', 'rf_recall', 'rf_f1score', 'rf_accuracy', 
                           'xgb_precision', 'xgb_recall', 'xgb_f1score', 'xgb_accuracy', 
                           'svc_precision', 'svc_recall', 'svc_f1score', 'svc_accuracy'])

for i in range(1,9):
    
    rslt= video_train_test_split(i, df_event_features)
    df_video.loc[len(df_video)] = rslt
    
    
df_video.to_csv('Individual_Video_test_results_transitional.csv')



