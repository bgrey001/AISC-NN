#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 18:35:11 2023
@author: benedict

Script for modelling the AIS data using a support vector machine classifier using tslearn library

"""
# =============================================================================
# dependencies
# =============================================================================
import numpy as np
import pickle
from datetime import datetime

import AIS_loader

from tslearn.utils import to_time_series_dataset # this pads with nans to create uniform sequence lengths
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC

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
    
    return to_time_series_dataset(features), targets


X_train, y_train = extract(train_seq_list)
X_test, y_test = extract(test_seq_list)



svc = TimeSeriesSVC(C=1.0, kernel='gak', gamma=0.1, n_jobs=-1)
knn = KNeighborsTimeSeriesClassifier(n_neighbors=2, metric='dtw', n_jobs=-1)
# model = knn
model = svc
        

def fit_predict(model, X_train, y_train, X_test, y_test, verbose=True):
    start_time = datetime.now()  
    model.fit(X_train, y_train)
    end_time = datetime.now()  
    if verbose: print(f'Fit time: {(end_time - start_time)}')

    # predict
    y_preds = model.predict(X_test)
    print(sum(y_preds==y_test)/len(y_test))
        
    confmat = ConfusionMatrix(actual_vector=y_test, predict_vector=y_preds)
    
    if verbose:
        confmat.print_matrix()
        confmat.stat(summary=True)
    
fit_predict(model, X_train, y_train, X_test, y_test)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
