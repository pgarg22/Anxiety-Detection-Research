#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 23:50:50 2023

@author: octopusphoenix
"""

from train_test_split_toolbox import random_split_test

from train_test_split_toolbox import all_participant_test,all_video_test

from train_test_split_toolbox import all_participant_transitional,all_video_transitional

import pandas as pd
import warnings
warnings.filterwarnings("ignore")




df_event_features= pd.read_csv("event_features_ecg_rsp.csv")

df_event_features_trans= pd.read_csv("event_features_ecg_rsp_transitional_phase.csv")


random_split_test(df_event_features,"norm")
random_split_test(df_event_features_trans,"trans")


all_participant_test(df_event_features)
all_video_test(df_event_features)


all_participant_transitional(df_event_features_trans)
all_video_transitional(df_event_features_trans)

