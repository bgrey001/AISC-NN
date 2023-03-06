#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 17:59:46 2023
@author: benedict


"""
# =============================================================================
# dependencies
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seglearn.base import TS_Data
from seglearn.datasets import load_watch
from seglearn.util import check_ts_data, ts_stats

import pickle
from datetime import datetime

import AIS_loader

# from tslearn.utils import to_time_series_dataset # this pads with nans to create uniform sequence lengths
# from tslearn.neighbors import KNeighborsTimeSeriesClassifier
# from tslearn.svm import TimeSeriesSVC

from pycm import ConfusionMatrix


# =============================================================================
# load data
# starting with the standardised unpadded varying time sequences 
# =============================================================================

choice ='varying'
version = '3'
def load_data(split):
    np.random.seed(15) # set random seed to reproduce random results
    with open(f'../../data/pkl/{choice}/{split}_v{version}.pkl', 'rb') as f:
        seq_list = pickle.load(f)
    with open(f'../../data/pkl/{choice}/utils_v{version}.pkl', 'rb') as f:
        obj = pickle.load(f)
        n_features = obj[0]
        n_classes = obj[1]
        seq_length = obj[2]
        
    return seq_list, n_features, n_classes, seq_length


train_seq_list, _, _, _ = load_data('test')
test_seq_list, _, _, _ = load_data('valid')



# extract targets and features from sequences in fn
def extract(seq_list):
    targets = []
    features = []
    idx = 0
    for seq in seq_list:
        features.append(seq[:, :-1])
        targets.append(seq[:, -1][0])
        idx += 1
        if idx==150:
            break
    
    return features, targets


X_train, y_train = extract(train_seq_list)
X_test, y_test = extract(test_seq_list)


X_TS = TS_Data(X_train)
















