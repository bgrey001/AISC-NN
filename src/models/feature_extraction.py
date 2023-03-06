#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:17:58 2023
@author: Benedict Grey

Feature extraction script using tsfresh library

"""

# =============================================================================
# dependencies
# =============================================================================
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from pycm import ConfusionMatrix

import AIS_loader

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tqdm import tqdm


# =============================================================================
# load data
# starting with the standardised unpadded varying time sequences 
# =============================================================================
choice ='varying'
version = '3'
def load_sequences(split):
    np.random.seed(15) # set random seed to reproduce random results
    with open(f'../../data/pkl/{choice}/{split}_v{version}.pkl', 'rb') as f:
        seq_list = pickle.load(f)
    with open(f'../../data/pkl/{choice}/utils_v{version}.pkl', 'rb') as f:
        obj = pickle.load(f)
        n_features = obj[0]
        n_classes = obj[1]
        seq_length = obj[2]
        
    return seq_list, n_features, n_classes, seq_length


train_seq_list, _, _, _ = load_sequences('train')
test_seq_list, _, _, _ = load_sequences('test')
valid_seq_list, _, _, _ = load_sequences('valid')
seq_list = train_seq_list + test_seq_list + valid_seq_list

# testing splits of data, didn't use stratified split due to size of data and the results suggest this was the right call, very even split
def test_sample_size(train_seq_list, test_seq_list, valid_seq_list):
    train_arr = np.concatenate(train_seq_list)
    df1 = pd.DataFrame(train_arr)
    res1 = df1[5].value_counts(normalize=True)
    
    test_arr = np.concatenate(test_seq_list)
    df2 = pd.DataFrame(test_arr)
    res2 = df2[5].value_counts(normalize=True)
    
    valid_arr = np.concatenate(valid_seq_list)
    df3 = pd.DataFrame(valid_arr)
    res3 = df3[5].value_counts(normalize=True)

# =============================================================================
# extract targets and features from sequences and then perform feature extraction using tsfresh
# =============================================================================
def process(seq_list, select, save, save_version):
    out_df = pd.DataFrame(columns = ['id','time','f1', 'f2', 'f3', 'f4', 'f5'])
    targets = []
    
    for idx, seq in enumerate(tqdm(seq_list)):
        seq_feat = seq[:, :-1]
        seq_targ = seq[:, -1][0]
        targets.append(seq_targ)
        
        sub_df = pd.DataFrame(seq_feat, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        sub_df.insert(loc=0, column='time', value=sub_df.index)
        sub_df.insert(loc=0, column='id', value=idx)
        
        out_df = pd.concat([out_df, sub_df])
        # if idx == 1500: break
        
    targets = pd.Series(targets)
    
    # extract features
    extraction_settings = ComprehensiveFCParameters() # this can be tuned?
    
    # each sequence is being pushed into a single row with all features extracted along the y axis
    X_fe = extract_features(out_df, column_id='id', column_sort='time', default_fc_parameters=extraction_settings, impute_function=impute)
    
    # feature selection
    if select:
        X_filt = select_features(X_fe, targets)
        X_filt.head()
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X_fe, targets, test_size=0.4)
    if select: X_train, X_test = X_train[X_filt.columns], X_test[X_filt.columns]
    
    if save:
        with open(f'../../data/pkl/feature_extraction/v{save_version}.pkl', 'wb') as f:
            pickle.dump([X_train, X_test, y_train, y_test], f)
        print(f'Saved v{save_version}')
    
    else: return out_df, targets.reshape(-1, 1)


# =============================================================================
# load data from pkl
# =============================================================================
def load_data(save_version):
    with open(f'../../data/pkl/feature_extraction/v{save_version}.pkl', 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
        
    return X_train, X_test, y_train, y_test


process(seq_list=seq_list, select=True, save=True, save_version=1)



# =============================================================================
# load data, fit and test
# =============================================================================
# X_train, X_test, y_train, y_test = load_data(save_version=1)
# svm_clf = SVC(C=1, kernel='rbf', verbose=1)
# svm_clf.fit(X_train, y_train)
# print(classification_report(y_test, svm_clf.predict(X_test)))





















