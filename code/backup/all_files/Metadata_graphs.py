#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 23:10:06 2022

@author: octopusphoenix
"""

from load_metadata import *
from graphs_toolbox import push_viz_scatter , push_viz_bar, push_viz_scatter_subplots,push_viz_bar_subplots



##############################################################################################


# Graph in order the data collected from participants(Participant ID)
push_viz_scatter_subplots(2, 
                 demographic_df.loc[:, ["Participant ID","Beck Anxiety", "Hamilton Anxiety"]],
                 "Anxiety Distributions",
                 ["Beck Anxiety", "Hamilton Anxiety"],
                 "Anxiety Level",
                 "markers+lines",
                 ["diamond","cross"],
                 [63,56],
                 [["Minimum(0-7)","Mild(8-15)","Moderate(16-25)","Severe(26-63)"],
                  ["Minimum(0-7)","Mild(8-14)","Moderate(15-23)","Severe(24-56)"]])


##############################################################################################



# Graph in order of Age Scatter'
push_viz_scatter_subplots(2, 
                 demographic_df.loc[:, ["Age","Beck Anxiety", "Hamilton Anxiety"]],
                 "Age",
                 ["Beck Anxiety", 
                  "Hamilton Anxiety"],
                 "Anxiety Level",
                 "markers",
                 ["diamond","cross"],
                 [63,56],
                 [["Minimum(0-7)","Mild(8-15)","Moderate(16-25)","Severe(26-63)"],
                  ["Minimum(0-7)","Mild(8-14)","Moderate(15-23)","Severe(24-56)"]])


##############################################################################################


# Graph in order of Age Bar '
push_viz_bar(2,
             demographic_df.loc[:, ["AgeRange","Beck Anxiety", "Hamilton Anxiety"]],
             "Age_bar",
             ["Beck Anxiety", "Hamilton Anxiety"],
             "Anxiety Level",
             [63,56],
             [["Minimum(0-7)","Mild(8-15)","Moderate(16-25)","Severe(26-63)"],
              ["Minimum(0-7)","Mild(8-14)","Moderate(15-23)","Severe(24-56)"]])


##############################################################################################



# Graph in order of Gender '
push_viz_scatter_subplots(2, 
                 demographic_df.loc[:, ["Gender","Beck Anxiety", "Hamilton Anxiety"]],
                 "Gender",
                 ["Beck Anxiety", "Hamilton Anxiety"],
                 "Anxiety Level",
                 "markers",
                 ["diamond","cross"],
                 [63,56],
                 [["Minimum(0-7)","Mild(8-15)","Moderate(16-25)","Severe(26-63)"],
                  ["Minimum(0-7)","Mild(8-14)","Moderate(15-23)","Severe(24-56)"]])


push_viz_bar_subplots(2, 
             demographic_df.loc[:, ["Gender","Beck Anxiety", "Hamilton Anxiety"]],
             "Gender_bar",
             ["Beck Anxiety", "Hamilton Anxiety"],
             "Anxiety Level",
             [63,56],
             [["Minimum(0-7)","Mild(8-15)","Moderate(16-25)","Severe(26-63)"],
             ["Minimum(0-7)","Mild(8-14)","Moderate(15-23)","Severe(24-56)"]])

##############################################################################################




# Graph in order of BMI '

push_viz_scatter_subplots(2, 
                 demographic_df.loc[:, ["BMI","Beck Anxiety", "Hamilton Anxiety"]],
                 "BMI",
                 ["Beck Anxiety", "Hamilton Anxiety"],
                 "Anxiety Level",
                 "markers",
                 ["diamond","cross"],
                 [63,56],
                 [["Minimum(0-7)","Mild(8-15)","Moderate(16-25)","Severe(26-63)"],
                  ["Minimum(0-7)","Mild(8-14)","Moderate(15-23)","Severe(24-56)"]])

push_viz_bar_subplots(2, 
             demographic_df.loc[:, ["BMI","Beck Anxiety", "Hamilton Anxiety"]],
             "BMI_bar",
             ["Beck Anxiety", "Hamilton Anxiety"],
             "Anxiety Level",
             [63,56],
             [["Minimum(0-7)","Mild(8-15)","Moderate(16-25)","Severe(26-63)"],
              ["Minimum(0-7)","Mild(8-14)","Moderate(15-23)","Severe(24-56)"]])
