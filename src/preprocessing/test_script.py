#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:51:08 2022

@author: BenedictGrey
"""

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pre_processing_module as ppm

max_seq_length = 510

t_model = ppm.pre_processing_module('trollers')

x, v, y = t_model.return_test_train_sets(max_seq_length) # all the training examples for all 5 ships in different sets



# for i, df_list in enumerate(seq):
#     # print(i)
#     for j, df in enumerate(df_list):
#         df_list[j] = df_list[j].reindex(list(range(0, max_seq_length))).reset_index(drop=True)
#         # break
    
    
# dataframe = seq[0][3]
# dataframe = dataframe.reindex(list(range(0, max_seq_length))).reset_index(drop=True)

    


# p_model = ppm.pre_processing_module('pole_and_line')


# list_df = t_model.partition_vessels() # all the different vessels in the dataset
# df = t_model.segment_dataframe(list_df[0], time_period=24, threshold=1) # picked one ship and partitioned them into 

# drop_columns = ['mmsi', 'timestamp', 'distance_from_shore', 'distance_from_port', 
#                 'delta_t', 'delta_t_cum', 'course', 'speed', ]

# # df = t_model.create_deltas(list_df, drop_columns)
# df = t_model.create_deltas(list_df=df, max_seq_length=max_seq_length, uniform=True, drop_columns=drop_columns)


# example = list_df[0]
# train, valid, test = t_model.return_test_train_sets()
# x,y,z = t_model.return_test_train_sets()
# print(z['delta_c'])


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# z[['lat', 'lon', 'delta_c', 'delta_s']] = scaler.fit_transform(z[['lat', 'lon', 'delta_c', 'delta_s']])
