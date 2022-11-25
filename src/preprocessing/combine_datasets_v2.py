#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:48:15 2022

@author: benedict

Combining datasets for varying time sequences

"""

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pre_processing_module as ppm1
import pre_processing_module_v2 as ppm2
import dask.dataframe as dd
import pickle
import random
import math

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler



# from dask_ml.model_selection import train_test_split

version = 1
choice = 'varying'

troller_module = ppm2.pre_processing_module('trollers')
pole_module = ppm2.pre_processing_module('pole_and_line') 
purse_module = ppm2.pre_processing_module('purse_seines') 
fixed_module = ppm2.pre_processing_module('fixed_gear') 
trawlers_module = ppm2.pre_processing_module('trawlers') 
drifting_module = ppm2.pre_processing_module('drifting_longlines') 


# =============================================================================
# varying sequence lengths, padding will take place in the data loader module
# =============================================================================

match choice: 
    case 'varying':
        
        time_period = 24
        threshold = 1
        saving_parquet = False
        saving = False
        
        aggregate_mask = []
        drop_columns = ['mmsi', 'distance_from_shore', 'distance_from_port', 'delta_t_cum', 'course', 'is_fishing', 'delta_t', 'desired', 'timestamp']
        
        # trollers, order matters here!
        troller_list_df = troller_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        troller_df, troller_mask = troller_module.flatten_df_list(troller_list_df)
        aggregate_mask += troller_mask
        aggregate_df = dd.from_pandas(troller_df, npartitions=1)
        
        # pole and line
        pole_list_df = pole_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        pole_df, pole_mask = pole_module.flatten_df_list(pole_list_df)
        aggregate_mask += pole_mask
        aggregate_df = dd.concat([aggregate_df, pole_df])
        
        # purse seines
        purse_list_df = purse_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        purse_df, purse_mask = purse_module.flatten_df_list(purse_list_df)
        aggregate_mask += purse_mask
        aggregate_df = dd.concat([aggregate_df, purse_df])
        
        # fixed gear
        fixed_list_df = fixed_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        fixed_df, fixed_mask = fixed_module.flatten_df_list(fixed_list_df)
        aggregate_mask += fixed_mask
        aggregate_df = dd.concat([aggregate_df, fixed_df])
        
        # trawlers
        trawlers_list_df = trawlers_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        trawlers_df, trawlers_mask = trawlers_module.flatten_df_list(trawlers_list_df)
        aggregate_mask += trawlers_mask
        aggregate_df = dd.concat([aggregate_df, trawlers_df])
        
        # drifting longlines
        drifting_list_df = drifting_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        drifting_df, drifting_mask = drifting_module.flatten_df_list(drifting_list_df)
        aggregate_mask += drifting_mask
        aggregate_df = dd.concat([aggregate_df, drifting_df])
        
        
        # =============================================================================
        # save files
        # =============================================================================
        if saving_parquet:
            aggregate_df.to_parquet(f"../../data/parquet/aggregate_v{version}", engine="pyarrow", write_index=False)
            with open(f'../../data/parquet/aggregate_mask_v{version}.pkl', 'wb') as f:
                pickle.dump(aggregate_mask, f)
    
        # =============================================================================
        # read files
        # =============================================================================
        else:
            ddf = dd.read_parquet(f"../../data/parquet/aggregate_v{version}", engine="pyarrow", index=False)
            with open(f'../../data/parquet/aggregate_mask_v{version}.pkl', 'rb') as f:
                aggregate_mask = pickle.load(f)
                
            aggregate_df = ddf.compute()
        
        # instantiate empty preprocessing module for the operations to follow
        module = ppm2.pre_processing_module("")
        
        # rebuild the sequences using the mask, then we shuffle and create more masks
        df_list = module.re_segment(aggregate_df, aggregate_mask, dataframe=True)
        
        # find max sequence length for padding process
        len_max_seq = max(df_list)
        break
        
        # now shuffle
        random.shuffle(df_list)
        
        # or do we split the data while it's batches?
        batch_train, batch_valid, batch_test = module.split(df_list)
        total = len(aggregate_df)
        
        # now we flatten the batches
        batch_train_flat, batch_train_mask = module.flatten_df_list(batch_train, nested_list=False)
        batch_valid_flat, batch_valid_mask = module.flatten_df_list(batch_valid, nested_list=False)
        batch_test_flat, batch_test_mask = module.flatten_df_list(batch_test, nested_list=False)
        # print(len(batch_train_flat) + len(batch_valid_flat) + len(batch_test_flat)) # only 3 off the original size of the data 
        
        # now we normalise
        scaler = StandardScaler()
        
        # covert dataframes to arrays
        data_train = batch_train_flat.values 
        data_valid = batch_valid_flat.values
        data_test = batch_test_flat.values
        
        # scale
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
        
        # counter = 0
        # for i in data_train_seqs:
        #     counter += len(i)
        # for i in data_valid_seqs:
        #     counter += len(i)
        # for i in data_test_seqs:
        #     counter += len(i)
        # print(total - counter) # it works!
        
        # =============================================================================
        # save
        # =============================================================================
        if saving:
            with open(f'../../data/pkl/{choice}/train_v{version}.pkl', 'wb') as f:
                pickle.dump(data_train_seqs, f)
            with open(f'../../data/pkl/{choice}/valid_v{version}.pkl', 'wb') as f:
                pickle.dump(data_valid_seqs, f)
            with open(f'../../data/pkl/{choice}/test_v{version}.pkl', 'wb') as f:
                pickle.dump(data_test_seqs, f)
            print('Saved files successfully')

