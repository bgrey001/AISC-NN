#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 18:12:58 2023
@author: benedict

Step by step:
    1) Import dataset
    2) Resample individual vessels to regular intervals at designated size
    3) Linearly interpolate the each vessel
    4) Concatenate vessel data into one dataset with all vessels
    5) Segment into 24 hours sequences
    6) Shuffle sequences
    7) Flatten sequences
    8) Split into train, test and validation sets
    9) Scale the data based on the values calculated on train data
    10) Segment again into 24 hours sequences, shuffle and export
    
"""


import pandas as pd
import datetime 
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pre_processing_module_v2 as ppm
import dask.dataframe as dd
import pickle
import random
import math
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

import pre_processing_module_v2 as ppm

random.seed(42)

# =============================================================================
# Import dataset
# =============================================================================
troller_module = ppm.pre_processing_module('trollers')
trawler_module = ppm.pre_processing_module('trawlers')


troller_vessel_list = troller_module.partition_vessels()
trawler_vessel_list = trawler_module.partition_vessels()

n_classes = 4
target_interval = 60 # just for testing keep this higher otherwise it will be expensive compute

# =============================================================================
# def linear_interpolation(vessel_list, visualise=False):
#     
#     drop_columns = ['timestamp', 'mmsi', 'distance_from_shore', 'distance_from_port', 'is_fishing']
# 
#     # prepare master dataframe
#     df_ex = vessel_list[0]
#     df = pd.DataFrame().reindex_like(df_ex).dropna()
#     df.drop(columns=drop_columns, inplace=True)
# 
#     df_orig = df.copy()
#     del df_ex # keep memory clean
# 
#     for v in vessel_list:
#         # =============================================================================
#         # Resample each vessel
#         # =============================================================================
#         v.index = v['timestamp'] # set index to timestamp
#         v.index.name = 'index'
#         v.drop(columns=drop_columns, inplace=True)
#         v = v[~v.index.duplicated(keep='first')]
#         resampled_v = v.resample(str(target_interval) + 'T').mean()
#         
#         # =============================================================================
#         # Linear interpolation
#         # =============================================================================
#         resampled_v = resampled_v.interpolate(method='linear')
#         
#         # =============================================================================
#         # Combine vessels into master dataframe
#         # =============================================================================
#         df = pd.concat([df, resampled_v])
#         df_orig = pd.concat([df_orig, v])
#         
#         # =============================================================================
#         # Visualise the difference between the intepolated and original trajectories
#         # =============================================================================
#         if visualise:
#             plt.plot(v['lat'], v['lon'])
#             plt.show()
#             plt.plot(resampled_v['lat'], resampled_v['lon'], c='green')
#             plt.show()
# 
#     return df
# 
# =============================================================================
module = ppm.pre_processing_module("")

# join different datasets together
df = pd.concat([
                module.linear_interpolation(troller_vessel_list),
                module.linear_interpolation(trawler_vessel_list)]).values

# =============================================================================
# Segment into 24 hours sequences
# 1440 minutes in a day, therefore the total length of the seqence should be 1440 / target_interval
# =============================================================================
seq_length = 1440 / target_interval
def linear_segment(data):
    return np.array_split(data, math.ceil(data.shape[0]/seq_length))

# convert to sequences
list_seq = linear_segment(df)
del df # clean

# shuffle
random.shuffle(list_seq)

# =============================================================================
# Split into train and test sets
# =============================================================================
data_train, data_valid, data_test = module.split(data=list_seq, train_ratio=0.8)
del list_seq # clean

# =============================================================================
# Flatten back into respective matrices
# =============================================================================
train_flat = np.vstack(data_train)
valid_flat = np.vstack(data_valid)
test_flat = np.vstack(data_test)
del data_train, data_valid, data_test # clean

# =============================================================================
# 6) Scale the data based on the values calculated on train data
# =============================================================================
# scaler = MinMaxScaler()
scaler = StandardScaler()

# scale
scale_train = scaler.fit_transform(train_flat[:, [0, 1, 2, 3]])
scale_valid = scaler.transform(valid_flat[:, [0, 1, 2, 3]])
scale_test = scaler.transform(test_flat[:, [0, 1, 2, 3]])

# add scaled columns back to data arrays
train_flat[:, [0, 1, 2, 3]] = scale_train
valid_flat[:, [0, 1, 2, 3]] = scale_valid
test_flat[:, [0, 1, 2, 3]] = scale_test

list_train = linear_segment(train_flat)
list_valid = linear_segment(valid_flat)
list_test = linear_segment(test_flat)




