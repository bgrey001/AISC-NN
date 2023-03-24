#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:07:12 2023

@author: benedict
"""

import pandas as pd
import pickle as pkl
import numpy as np
import pickle
from datetime import datetime

import dask
from dask.dataframe import from_pandas

import AIS_loader

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tqdm import tqdm


with open('../../data/pkl/feature_extraction/dfs_v1.pkl', 'rb') as f:
    features, targets = pkl.load(f)
    

ddf = from_pandas(features, npartitions=6)
    
    
# extract features
extraction_settings = MinimalFCParameters() # this can be tuned?

# each sequence is being pushed into a single row with all features extracted along the y axis
X_fe = extract_features(ddf, column_id='id', column_sort='time', default_fc_parameters=extraction_settings, disable_progressbar=False)
X_fe.to_csv('../../data/csv/fe.csv')
df = X_fe.compute()
df['Y'] = targets
df.to_csv('../../data/csv/fe_pd.csv')


