#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 01:12:47 2022

@author: benedict

Script to process the data using the pre_processing_module into train test and validataion sets for all the datasets and combine them into 3 large sets:
training validation and testing data and then export using pickle ready for loading in the network scripts


"""

import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.preprocessing import StandardScaler

import pre_processing_module as ppm

"""
IMPORTANT: Need to set the number of sequences to be even across the classes so they don't introduce bias into the trianing algorithms
v1 DATA:
        
    Number of training sequences for two smallest datasets:
        pole_and_line = 2192
        trollers = 1700
        
    Number of validation sequences for two smallest dataest:
        pole_and_line = 228
        trollers = 217
        
    Number of test sequences for two smallest dataest:
        pole_and_line = 240
        trollers = 252
        
v2 DATA -> no threshold
    Number of training sequences for two smallest datasets:
        pole_and_line = 2498
        trollers = 1742
        
    Number of validation sequences for two smallest dataest:
        pole_and_line = 244
        trollers = 220
        
    Number of test sequences for two smallest dataest:
        pole_and_line = 286
        trollers = 266
        
        
v3 DATA -> no datasize reduction, all data being preserved

"""


options = ['varying', 'padded', 'linear_interp']
choice = 'linear_interp' # set choice
version_num = 2 # version for filing system


# instantiate ppm modules
trollers_module = ppm.pre_processing_module('trollers')
pole_and_line_module = ppm.pre_processing_module('pole_and_line')
purse_seine_module = ppm.pre_processing_module('purse_seines')
fixed_gear_module = ppm.pre_processing_module('fixed_gear')
trawlers_module = ppm.pre_processing_module('trawlers')
drifting_longlines_module = ppm.pre_processing_module('drifting_longlines')


if choice == 'varying':
    print('Creating varying sequence sizes...')
    trollers_list_train_seq, trollers_list_valid_seq, trollers_list_test_seq = trollers_module.return_test_train_sets(uniform=False) # varying
    pole_and_line_list_train_seq, pole_and_line_list_valid_seq, pole_and_line_list_test_seq = pole_and_line_module.return_test_train_sets(uniform=False)
    purse_seine_list_train_seq, purse_seine_list_valid_seq, purse_seine_list_test_seq = purse_seine_module.return_test_train_sets(uniform=False)
    fixed_gear_list_train_seq, fixed_gear_list_valid_seq, fixed_gear_list_test_seq = fixed_gear_module.return_test_train_sets(uniform=False)
    trawlers_list_train_seq, trawlers_list_valid_seq, trawlers_list_test_seq = trawlers_module.return_test_train_sets(uniform=False)
    drifting_longlines_list_train_seq, drifting_longlines_list_valid_seq, drifting_longlines_list_test_seq = drifting_longlines_module.return_test_train_sets(uniform=False)

    
    
if choice == 'uniform':
    print('Creating uniform batch sizes...')
    trollers_list_train_seq, trollers_list_valid_seq, trollers_list_test_seq = trollers_module.return_test_train_sets(uniform=True) # uniform
    pole_and_line_list_train_seq, pole_and_line_list_valid_seq, pole_and_line_list_test_seq = pole_and_line_module.return_test_train_sets(uniform=True)
    purse_seine_list_train_seq, purse_seine_list_valid_seq, purse_seine_list_test_seq = purse_seine_module.return_test_train_sets(uniform=True)
    fixed_gear_list_train_seq, fixed_gear_list_valid_seq, fixed_gear_list_test_seq = fixed_gear_module.return_test_train_sets(uniform=True)
    trawlers_list_train_seq, trawlers_list_valid_seq, trawlers_list_test_seq = trawlers_module.return_test_train_sets(uniform=True)
    drifting_longlines_list_train_seq, drifting_longlines_list_valid_seq, drifting_longlines_list_test_seq = drifting_longlines_module.return_test_train_sets(uniform=True)


if choice == 'linear_interp': # linear interpolation
    print('Linear interpolation beginning...')
    trollers_list_train_seq, trollers_list_valid_seq, trollers_list_test_seq = trollers_module.return_test_train_sets(uniform=False, interpolated=True) # linearly interpolated
    pole_and_line_list_train_seq, pole_and_line_list_valid_seq, pole_and_line_list_test_seq = pole_and_line_module.return_test_train_sets(uniform=False, interpolated=True)
    purse_seine_list_train_seq, purse_seine_list_valid_seq, purse_seine_list_test_seq = purse_seine_module.return_test_train_sets(uniform=False, interpolated=True)
    fixed_gear_list_train_seq, fixed_gear_list_valid_seq, fixed_gear_list_test_seq = fixed_gear_module.return_test_train_sets(uniform=False, interpolated=True)
    trawlers_list_train_seq, trawlers_list_valid_seq, trawlers_list_test_seq = trawlers_module.return_test_train_sets(uniform=False, interpolated=True)
    drifting_longlines_list_train_seq, drifting_longlines_list_valid_seq, drifting_longlines_list_test_seq = drifting_longlines_module.return_test_train_sets(uniform=False, interpolated=True)
    





# =============================================================================
# combine lists and save as np files
# =============================================================================
agg_list_train_seq = trollers_list_train_seq + pole_and_line_list_train_seq + purse_seine_list_train_seq + fixed_gear_list_train_seq + trawlers_list_train_seq + drifting_longlines_list_train_seq
agg_list_valid_seq = trollers_list_valid_seq + pole_and_line_list_valid_seq + purse_seine_list_valid_seq + fixed_gear_list_valid_seq + trawlers_list_valid_seq + drifting_longlines_list_valid_seq
agg_list_test_seq = trollers_list_test_seq + pole_and_line_list_test_seq + purse_seine_list_test_seq + fixed_gear_list_test_seq + trawlers_list_test_seq + drifting_longlines_list_test_seq




if choice == 'linear_interp':
    # they all need scaling together
    def split_scale():
        
        batch_size = len(agg_list_valid_seq[0])
        
        # aggregate
        main_train = np.concatenate(agg_list_train_seq)
        main_valid = np.concatenate(agg_list_valid_seq)
        main_test = np.concatenate(agg_list_test_seq)
        
        # scale
        scaler = StandardScaler()
        main_train[:, :-1] = scaler.fit_transform(main_train[:, :-1])
        main_valid[:, :-1] = scaler.transform(main_valid[:, :-1])
        main_test[:, :-1] = scaler.transform(main_test[:, :-1])
        
        # rebatch and convert nans
        main_train = np.nan_to_num(main_train)
        main_valid = np.nan_to_num(main_valid)
        main_test = np.nan_to_num(main_test)
        
        main_train = np.split(main_train, len(main_train)/batch_size)
        main_valid = np.split(main_valid, len(main_valid)/batch_size)
        main_test = np.split(main_test, len(main_test)/batch_size)
        
        return main_train, main_valid, main_test
        
    agg_list_train_seq, agg_list_valid_seq, agg_list_test_seq = split_scale()




if (choice == 'varying'):
    max_seq_length = 0
    for item in agg_list_train_seq:
        if max_seq_length < len(item):
            max_seq_length = len(item)
    print(f'max sequence length is {max_seq_length}')
        

# save
with open(f'../../data/pkl/{choice}/train_v{version_num}.pkl', 'wb') as f:
    pickle.dump(agg_list_train_seq, f)
with open(f'../../data/pkl/{choice}/valid_v{version_num}.pkl', 'wb') as f:
    pickle.dump(agg_list_valid_seq, f)
with open(f'../../data/pkl/{choice}/test_v{version_num}.pkl', 'wb') as f:
    pickle.dump(agg_list_test_seq, f)
    
print(f'Saved files successfully')
