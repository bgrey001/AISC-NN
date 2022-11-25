#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:24:56 2022

@author: benedict

Data preprocessing to organise uninteropalated data into sequences of varying lengths for time periods of 24hrs

TODO:
    Split the data sets into vessel subsets
    Take a percentage of each vessel subset as the training, say 20%
    Normalise each training set using the scaler fitted on training data for each vessel
    Fit the scaler to test data for each vessel

"""

import pandas as pd
import numpy as np
import torch
import math

import matplotlib.pyplot as plt


import datetime as dt
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# from utils_1 import ALL_LETTERS, N_LETTERS # attributes, or variables
# from utils_1 import load_data, letter_to_tensor, line_to_tensor, random_training_example # methods or functions


import pre_processing_module as ppm

module = ppm.pre_processing_module('trollers')

df = module.df
list_df = module.partition_vessels()
sequences = module.build_sequences(list_df=list_df, time_period=24, threshold=4, normalise=True)


# item = sequences[0]
# print(item[:, 2])

# NORMALISE THE DATA, MinMaxScaler works on both 




# Split the data sets into vessel subsets

df = module.partition_vessels()
dfx = df.iloc[:, :-1]
dfy = df.iloc[:, -1]



def split_scale(list_df): # takes as input the output of partition_vessels()
    list_train = []
    list_test = []
    for df in list_df:
        dfx = df.iloc[:, :-1]
        dfy = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=None, shuffle=False, stratify=None)
        scaler = MinMaxScaler()
        X_train[['speed', 'course', 'lat', 'lon']] = scaler.fit_transform(X_train[['speed', 'course', 'lat', 'lon']])
        X_test[['speed', 'course', 'lat', 'lon']] = scaler.transform(X_test[['speed', 'course', 'lat', 'lon']])
        df_train = X_train
        df_train['desired'] = y_train
        df_test = X_test
        df_test['desired'] = y_test
        list_train.append(df_train)
        list_test.append(df_test)
    return list_train, list_test
        
        

    
ltrain, ltest = split_scale(df)

ltrain_seq = module.build_sequences(list_df=ltrain, time_period=24, threshold=4, normalise=False)
ltest_seq = module.build_sequences(list_df=ltest, time_period=24, threshold=4, normalise=False)






# =============================================================================
# def normalise_sequences(sequences):
#     for i in range(len(sequences)):
#         scaler = MinMaxScaler()
#         features = sequences[i][:, :-1]
#         sequences[i][:, :-1] = scaler.fit_transform(features)
# 
# 
# 
# 
# normalise_sequences(sequences2)
# 
# =============================================================================
# scaler = MinMaxScaler()

# X = item[:, :-1]
# y = item[:, -1]

# scaler = scaler.fit_transform(X)

# scaler2 = MinMaxScaler()

# X2 = item[:, :4]
# scaled2 = scaler2.fit_transform(X2)



# # item[:, :-1] = 
# A = math.cos(item[:, 2]) * math.cos(item[:, 3]) 


# a = item[:, 2]
















# class 

def segment_dataframe(dataframe, time_period, threshold): # must be run on unique dataframes
    
    # def compare_mmsi(index_1, index_2):
    #     if dataframe.iloc[index_1]['mmsi'] == dataframe.iloc[index_1]['mmsi']:
    #         return False
    #     else:
    #         return True
    
    # list to be returned
    list_df = []
    
    # set first timestamp and the time difference threshold
    t0 = dataframe['timestamp'].iloc[0]
    delta_size = timedelta(hours=time_period)
    curr_dif = timedelta(hours=0)

    # temp make the df smaller for testing
    # dataframe = dataframe.head(50)
    initial_index = 0

    for i in range(0, len(dataframe)):
        
        curr_t = dataframe['timestamp'].iloc[i]
        curr_dif = curr_t - t0 # difference from current time to first time

        # need to make sure time difference is >= time difference and that the ship's id is the same
        if (curr_dif >= delta_size):
            # make sure size is meaningful (greater than threshold)
            if ((i - initial_index) > threshold):     
                list_df.append(dataframe.iloc[initial_index : i, :])
            t0 = dataframe['timestamp'].iloc[i]
            initial_index = i

    return list_df


def create_deltas(list_df): # also removing some undesirable columns, should refactor
    total_num_seconds = 86400
    new_list = []
    for df in list_df:
        # compute change in time for each timestep and create feature delta_t
        # df['timestamp'] = df.index
        df['delta_t'] = (df['timestamp']-df['timestamp'].shift()).fillna(pd.Timedelta(seconds=0))
        # df = df.drop(columns=['timestamp_1'])
        # get cummalative sum of the delta column
        df['delta_t_cum'] = df['delta_t'].cumsum()
        # print(type(df['delta_t']))
        df['normalised_delta'] = df['delta_t'].dt.total_seconds() / total_num_seconds # normalising date time
        df = df.drop(columns=['mmsi', 'timestamp', 'distance_from_shore', 'distance_from_port', 'is_fishing', 'delta_t', 'delta_t_cum'])
        
        # remove targets and add onto end
        
        
        new_list.append(df)
    return new_list




"""
We want shape: (seq_length, 1, num_features)
"""

# =============================================================================
# def build_sequences():
#         
#     df_list_seq = []
#     agg_seq = []
#     
#     # first, segment the datframes
#     for df in list_df:    
#         df_list_seq.append(segment_dataframe(dataframe=df, time_period=24, threshold=4))
#     
#     
#     for sub_list in df_list_seq:
#         agg_seq.append(create_deltas(sub_list))
#         
#     flat_list = [item for sublist in agg_seq for item in sublist]
#     
#     for i in range(len(flat_list)):
#         flat_list[i] = flat_list[i].to_numpy()
#     
# =============================================================================





    
# =============================================================================
# num_features = 12
# 
# 
# 
# x = flat_list[0]
# 
# y = np.expand_dims(x, axis=1)
# 
# 
# 
# tensor = torch.zeros(len(x), 1, num_features)
# for j , item in enumerate(x):
#     print(item)
#     tensor[j][0][item]
#     
# print(tensor)
# 
# 
#     
# for i in range(1):
#     # tensor = torch.zeros(len(flat_list[i]))
#     seq_length = len(flat_list[i])
#     tensor = torch.zeros(seq_length, 1, num_features)
#     for j, item in enumerate(flat_list[i]):
#         tensor[j][0][]
# 
# =============================================================================








# =============================================================================
# def seq_to_tensor(seq):
#     tensor = torch.zeros(len(line), 1, N_LETTERS) 
#     for i, letter in enumerate(line): # the enumerate() function adds a counter as the key of tghe enumerate object (for iterating through tuples when access to an index is needed)
#         tensor[i][0][letter_to_index(letter)] = 1
#     return tensor
# =============================================================================







# def line_to_tensor(line):
#     tensor = torch.zeros(len(line), 1, N_LETTERS)
#     for i, letter in enumerate(line): # the enumerate() function adds a counter as the key of tghe enumerate object (for iterating through tuples when access to an index is needed)
#         tensor[i][0][letter_to_index(letter)] = 1
#     return tensor




# list1 = segment_dataframe(df1, 24, 0)
# ex = list1[3]
# n_features = 12
# =============================================================================
# category_lines, all_categories = load_data()
# n_categories = len(all_categories) # number of classes for this classification task which is 18
# n_hidden = 128 # hyperparameter to be tuned
# input_tensor = line_to_tensor('Albert')
# print(input_tensor.size())
# print(input_tensor[0]) # this is 'A'
# category, line, category_tensor, line_tensor = random_training_example(category_lines=category_lines, all_categories=all_categories)
# print(line)
# print(line_tensor[0].size())
# arr = line_tensor.detach().numpy()
# print(arr)
# =============================================================================










