#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 20:21:47 2023

@author: octopusphoenix
"""




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest



from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


from functions_toolbox import transform_categorical


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from functions_toolbox import feature_selector_basic

from sklearn.ensemble import ExtraTreesClassifier

import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

from sklearn.feature_selection import mutual_info_classif
    
from sklearn.preprocessing import StandardScaler

df_event_features= pd.read_csv("event_features_ecg_rsp.csv")

X=df_event_features.loc[:, ~df_event_features.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
y=df_event_features['Condition']  #

def pca_selector(X, y,num_feats):
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)
    pca = PCA(n_components=30)
    fit = pca.fit(X_std)
    # summarize components
    print("Explained Variance: %s" % fit.explained_variance_ratio_.sum())
    list1= fit.components_
    
    # #res = list(map(abs, list1))
    
    # pca_info = pd.Series(list1)
    # pca_info.index = X.columns
    # pca_info.sort_values(ascending=False)
    # feature_name = X.columns.tolist()
    
    # selected_feature=pca_info.keys().to_list()[0:num_feats]
    
    # pca_support = [True if i in selected_feature else False for i in feature_name]
    
    # return(pca_support,selected_feature)


# pca_selector(X,y,10)
# support, features= pca_selector(X,y,10)

# # print(features)


num_feats=10
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)
pca = PCA(n_components=(2*num_feats))
fit = pca.fit(X_std)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_.sum())
list1= fit.components_
list2= list1.transpose()

columns=[]
for i in range(1,((2*num_feats)+1)):
    columns.append("Component "+ str(i))

df_components = pd.DataFrame(list2, index=X.columns,columns=columns)
df_components_abs= df_components.abs()
feat_pca= df_components_abs.idxmax().to_list()
features_list= list(dict.fromkeys(feat_pca))
selected_feature= features_list[0:num_feats]
feature_name = X.columns.tolist()
pca_support = [True if i in selected_feature else False for i in feature_name]

    # fa = FactorAnalyzer((2*num_feats), rotation="varimax", method='minres', use_smc=True)
    # fa.fit(X_std)
    
    # columns=[]
    # for i in range(1,((2*num_feats)+1)):
    #     columns.append("Factor "+ str(i))
    
    # loadings = pd.DataFrame(fa.loadings_, 
    #                         columns=columns, index=X.columns)
    # loadings_abs= loadings.abs()
    # feat_load= loadings_abs.idxmax().to_list()
    # features_list= list(dict.fromkeys(feat_load))
    # selected_feature= features_list[0:num_feats]
    # feature_name = X.columns.tolist()
    # fa_support = [True if i in selected_feature else False for i in feature_name]
    
    # return(fa_support,selected_feature)

