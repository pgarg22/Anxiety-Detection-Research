#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:12:15 2023

@author: octopusphoenix
"""


from graphs_toolbox import push_viz_scatter
import pandas as pd

df_participant= pd.read_csv('Individual_Participant_test_results.csv')

# Graph in order the data collected from participants(Participant ID)
push_viz_scatter(2, 
                 df_participant.loc[:, ["Unnamed: 0","rf_precision", "rf_recall"]],
                 "Prediction Distributions",
                 ["rf_precision", "rf_recall"],
                 "Correct prediction percentage",
                 "markers+lines",
                 ["diamond","cross"],
                 [100,100],
                 [["Minimum(0-7)","Mild(8-15)","Moderate(16-25)","Severe(26-63)"],
                  ["Minimum(0-7)","Mild(8-14)","Moderate(15-23)","Severe(24-56)"]])