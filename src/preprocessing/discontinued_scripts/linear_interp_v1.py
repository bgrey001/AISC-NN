#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:45:37 2023

@author: benedict

Linear interpolation test script


Step by step:
    1) Import dataset
    2) Resample individual vessels to regular intervals at designated size
    3) Linearly interpolate the each vessel
    4) Concatenate vessel data into one dataset with all vessels
    5) Split into train, test and validation sets
    5) Scale the data based on the values calculated on train data
    6) Segment into 24 hours sequences and export
    
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



n_classes = 4


# =============================================================================
# logic for linear interpolation: 
#     1) segment dataset according to MMSI
#     2) create reference time points for each vessel at 1 minute intervals to start (we need these for mTAN)
#     3) interpolate reference time points and remove original data
# =============================================================================

troller_module = ppm.pre_processing_module('trollers')

vessel_list = troller_module.partition_vessels()


list_output = []
target_interval = 60 # just for testing keep this higher otherwise it will be expensive compute

item = vessel_list[3].head(500)
drop_columns = ['mmsi', 'distance_from_shore', 'distance_from_port', 'is_fishing']
item.drop(columns=drop_columns, inplace=True)

ref_timepoints = item.copy()
real_timepoints = item.copy()
v = item.copy()

v.index = v['timestamp']
v.index.name = 'index'

ref_timepoints.index = ref_timepoints['timestamp']
ref_timepoints.index.name = 'index'

# working method using original technique
# invert boolean series to remove duplicates of timestamps
# resampled_timepoints = v[~v.index.duplicated(keep='first')]
# # upsample the dataframe using the mean dispatching function
# resampled_timepoints = resampled_timepoints.resample(str(target_interval) + 'T').mean()
# # use linear interpolation to fill the gaps in the data
# resampled_timepoints = resampled_timepoints.interpolate(method='linear')






# =============================================================================
# create reference timepoints
# =============================================================================
# # invert boolean series to remove duplicates of timestamps
ref_timepoints = ref_timepoints[~ref_timepoints.index.duplicated(keep='first')]
# # upsample the dataframe using the mean dispatching function
ref_timepoints = ref_timepoints.resample(str(target_interval) + 'T').mean()
# check if nan values add a label to indicate the item is a superficial observation
ref_timepoints['superficial'] = 0

# v.loc[v['desired'].isnull(), 'superficial'] = 1
ref_timepoints.iloc[:, :] = 0
# v = v.interpolate(method='linear')


# range of dates

# incorperate the original timepoints with the new reference time points and interpolate before removing

# plt.plot(ref_timepoints['lat'], ref_timepoints['lon'], c='orange')
# plt.show()
# plt.plot(real_timepoints['lat'], real_timepoints['lon'], c='black')
# plt.show()



# =============================================================================
# function to loop through each vessel in a dataset and add to a csv to help manage memory
# =============================================================================



for v in vessel_list:
    
    v.index = v['timestamp'] # set index to timestamp
    v.index.name = 'index'
    
    v = v[~v.index.duplicated(keep='first')]
    resampled_v = v.resample(str(target_interval) + 'T').mean()
    resampled_v = resampled_v.interpolate(method='linear')
    
    break
    # return resampled_v
















