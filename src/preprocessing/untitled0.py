#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:36:12 2022

@author: benedict
"""

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pre_processing_module_v2 as ppm
import dask.dataframe as dd
import pickle
import random
import math

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

choice = 'varying'
n_classes = 6

troller_module = ppm.pre_processing_module('trollers')
pole_module = ppm.pre_processing_module('pole_and_line') 


load_version = 1
save_version = 3
n_features = 5
time_period = 24
threshold = 1
saving_parquet = False
saving = False

aggregate_mask = []
drop_columns = ['mmsi', 'distance_from_shore', 'distance_from_port', 'delta_t_cum', 'course', 'is_fishing', 'delta_t', 'desired', 'timestamp']

# trollers, order matters here!
troller_list_df = troller_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
troller_df, troller_mask = troller_module.flatten_df_list(troller_list_df, nested_list=True)
aggregate_mask += troller_mask
# aggregate_df = dd.from_pandas(troller_df, npartitions=1)
aggregate_df = troller_df

# # pole and line
pole_list_df = pole_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
pole_df, pole_mask = pole_module.flatten_df_list(pole_list_df, nested_list=True)
aggregate_mask += pole_mask
aggregate_df = pd.concat([aggregate_df, pole_df], ignore_index=True, axis=0)



# df = aggregate_df.compute()


single = troller_list_df[0][0]
single_1 = aggregate_df.iloc[:2, :]

# troller_list_df = troller_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
# troller_df, troller_mask = troller_module.flatten_df_list(troller_list_df, nested_list=True)
# aggregate_mask += troller_mask
# # aggregate_df = dd.from_pandas(troller_df, npartitions=1)
# aggregate_df = troller_df

# # # pole and line
# pole_list_df = pole_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
# pole_df, pole_mask = pole_module.flatten_df_list(pole_list_df, nested_list=True)
# aggregate_mask += pole_mask
# aggregate_df = pd.concat([aggregate_df, pole_df],  ignore_index=True, axis=0)


# module = ppm.pre_processing_module("")


# df = aggregate_df


# df_list = module.re_segment(df, aggregate_mask, dataframe=True)
# # list_df = module.re_segment(df=troller_df, mask=troller_mask, dataframe=True)




# plt.plot(single['lat'], single['lon'])
# plt.plot(single_1['lat'], single_1['lon'])


