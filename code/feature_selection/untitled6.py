#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:22:52 2023

@author: octopusphoenix
"""

import pandas as pd

df_features= pd.read_csv("all_feature_selection_methods_results.csv")
features= df_features.head(10)["Feature"].to_list()