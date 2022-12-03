#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:42:53 2022

@author: BenedictGrey

Script to visualise the trajectories

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

import pre_processing_module_v2 as ppm
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler


troller_module = ppm.pre_processing_module('trollers')
drop_columns = ['mmsi', 'distance_from_shore', 'distance_from_port', 'delta_t_cum', 'course', 'is_fishing', 'delta_t', 'desired', 'timestamp']
troller_list_df = troller_module.build_sequences(time_period=24, threshold=1, drop_columns=drop_columns)
troller_df, troller_mask = troller_module.flatten_df_list(troller_list_df, nested_list=True)

def vis_raw_dataframe(list_df_nested):
    for list_ in list_df_nested:
        for i, day in enumerate(list_):
            print(day['lat'], day['lon'])
            plt.title(f'index: {i}, length of sequence: {len(day)} unnormalised')
            plt.plot(day['lat'], day['lon'])
            plt.scatter(day['lat'], day['lon'], s=15)
            plt.show()
# vis_raw_dataframe(troller_list_df)




# instantiate empty preprocessing module for the operations to follow
module = ppm.pre_processing_module("")

# rebuild the sequences using the mask, then we shuffle and create more masks
df_list = module.re_segment(troller_df, troller_mask, dataframe=True)


# find max sequence length for padding process
# seq_length = 0
# for item in df_list:
#     if len(item) > seq_length:
#         seq_length = len(item) # RESULT: 2931
        
        
    

# now shuffle
# random.seed(15)
# random.shuffle(df_list)

# split the data while it's in batches
batch_train, batch_valid, batch_test = module.split(df_list)
# total = len(aggregate_df)

# now we flatten the batches
batch_train_flat, batch_train_mask = module.flatten_df_list(batch_train, nested_list=False)
batch_valid_flat, batch_valid_mask = module.flatten_df_list(batch_valid, nested_list=False)
batch_test_flat, batch_test_mask = module.flatten_df_list(batch_test, nested_list=False)
# print(len(batch_train_flat) + len(batch_valid_flat) + len(batch_test_flat)) # only 3 off the original size of the data 

# covert dataframes to arrays
data_train = batch_train_flat.values 
data_valid = batch_valid_flat.values
data_test = batch_test_flat.values

# scale
scaler = StandardScaler()
norm_train = scaler.fit_transform(data_train[:, [0, 1, 2, 4]])
norm_valid = scaler.transform(data_valid[:, [0, 1, 2, 4]])
norm_test = scaler.transform(data_test[:, [0, 1, 2, 4]])
# print(len(norm_train) + len(norm_valid) + len(norm_test)) # only 3 off the original size of the data 

# convert nans to prevent vanishing gradient
norm_train = np.nan_to_num(norm_train)
norm_valid = np.nan_to_num(norm_valid)
norm_test = np.nan_to_num(norm_test)

# add normalised columns back to data arrays
data_train[:, [0, 1, 2, 4]] = norm_train
data_valid[:, [0, 1, 2, 4]] = norm_valid
data_test[:, [0, 1, 2, 4]] = norm_test
# print(len(data_train) + len(data_valid) + len(data_test)) # only 3 off the original size of the data 

# rebuild time sequences
data_train_seqs = module.re_segment(data_train, batch_train_mask, dataframe=False)
data_valid_seqs = module.re_segment(data_valid, batch_valid_mask, dataframe=False)
data_test_seqs = module.re_segment(data_test, batch_test_mask, dataframe=False)



# take second half of list
idx = math.floor(len(df_list) * 0.8)

df_sublist = df_list[idx: len(df_list)]
def vis_un_normalised(list_df):
# vessel_1 = data_train_seqs[0]
    for i, day in enumerate(list_df):
        print(day.iloc[:, [1, 2]])
        plt.title(f'index: {i}, length of sequence: {len(day)}, unnormalised')
        plt.plot(day['lat'], day['lon'])
        plt.scatter(day['lat'], day['lon'], s=15)
        plt.show()
        # if i == 5:
        #     break


def vis_normalised(list_seq):
    for i, seq in enumerate(list_seq):
            print(seq[:, 1], seq[:, 2])
            plt.title(f'index: {i}, length of sequence: {len(seq)}, normalised')
            plt.plot(seq[:, 1], seq[:, 2])
            plt.scatter(seq[:, 1], seq[:, 2], s=15)
            plt.show()
            # if i == 5:
            #     break
            
            
            
single = df_sublist[27]
coords = single.iloc[:, [1,2]]
coords.to_csv('../../plots/trajectories/trollers/csv/ex_2.csv')
s = single.to_numpy()

# plt.plot(coords['lat'], coords['lon'])
# plt.scatter(coords['lat'], coords['lon'], s=15)


# vis_un_normalised(df_sublist)



# we have established the plots are the same using these vis functions
# vis_normalised(data_train_seqs)

    
     
     