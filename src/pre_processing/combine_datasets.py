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
        
Minimum value chosen for each

New technique is to use ALL the data, instead of producing a balanced dataset

FOR 1DCNN: Need to create even sequence sizes to serve as input for the 1DCNN. 
    Also, batching should be introduced 





"""
train_max_length = 1742
valid_max_length = 220
test_max_length = 266

max_seq_length = 0

# =============================================================================
# trolers
# =============================================================================
trollers_module = ppm.pre_processing_module('trollers')
trollers_list_train_seq, trollers_list_valid_seq, trollers_list_test_seq = trollers_module.return_test_train_sets(train_max_length, valid_max_length, test_max_length)

for l in trollers_list_train_seq: # finding the max sequence length for each training sequence
    if (max_seq_length < len(l)):
        max_seq_length = len(l)
for l in trollers_list_valid_seq: # finding the max sequence length for each training sequence
    if (max_seq_length < len(l)):
        max_seq_length = len(l)
for l in trollers_list_test_seq: # finding the max sequence length for each training sequence
    if (max_seq_length < len(l)):
        max_seq_length = len(l)

# =============================================================================
# pole and line
# =============================================================================
pole_and_line_module = ppm.pre_processing_module('pole_and_line')
pole_and_line_list_train_seq, pole_and_line_list_valid_seq, pole_and_line_list_test_seq = pole_and_line_module.return_test_train_sets(train_max_length, valid_max_length, test_max_length)
for l in pole_and_line_list_train_seq: # finding the max sequence length for each training sequence
    if (max_seq_length < len(l)):
        max_seq_length = len(l)
for l in pole_and_line_list_valid_seq: # finding the max sequence length for each training sequence
    if (max_seq_length < len(l)):
        max_seq_length = len(l)
for l in pole_and_line_list_test_seq: # finding the max sequence length for each training sequence
    if (max_seq_length < len(l)):
        max_seq_length = len(l)



# =============================================================================
# purse seines
# =============================================================================
purse_seine_module = ppm.pre_processing_module('purse_seines')
purse_seine_list_train_seq, purse_seine_list_valid_seq, purse_seine_list_test_seq = purse_seine_module.return_test_train_sets(train_max_length, valid_max_length, test_max_length)



# =============================================================================
# fixed gear
# =============================================================================
fixed_gear_module = ppm.pre_processing_module('fixed_gear')
fixed_gear_list_train_seq, fixed_gear_list_valid_seq, fixed_gear_list_test_seq = fixed_gear_module.return_test_train_sets(train_max_length, valid_max_length, test_max_length)



# =============================================================================
# trawlers
# =============================================================================
trawlers_module = ppm.pre_processing_module('trawlers')
trawlers_list_train_seq, trawlers_list_valid_seq, trawlers_list_test_seq = trawlers_module.return_test_train_sets(train_max_length, valid_max_length, test_max_length)


# =============================================================================
# drifting longlines
# =============================================================================
drifting_longlines_module = ppm.pre_processing_module('drifting_longlines')
drifting_longlines_list_train_seq, drifting_longlines_list_valid_seq, drifting_longlines_list_test_seq = drifting_longlines_module.return_test_train_sets(train_max_length, valid_max_length, test_max_length)

# =============================================================================
# combine lists and save as np files
# =============================================================================
agg_list_train_seq = trollers_list_train_seq + pole_and_line_list_train_seq + purse_seine_list_train_seq + fixed_gear_list_train_seq + trawlers_list_train_seq + drifting_longlines_list_train_seq
agg_list_valid_seq = trollers_list_valid_seq + pole_and_line_list_valid_seq + purse_seine_list_valid_seq + fixed_gear_list_valid_seq + trawlers_list_valid_seq + drifting_longlines_list_valid_seq
agg_list_test_seq = trollers_list_test_seq + pole_and_line_list_test_seq + purse_seine_list_test_seq + fixed_gear_list_test_seq + trawlers_list_test_seq + drifting_longlines_list_test_seq




# =============================================================================
# Max sequence for batching when uniform batches are required as input for a network
# max_seq_length will serve as the sequence size for ALL, before batching
# =============================================================================
for l in agg_list_train_seq: # finding the max sequence length for each training sequence
    if (max_seq_length < len(l)):
        max_seq_length = len(l)
for l in agg_list_valid_seq: # finding the max sequence length for each test sequence
    if (max_seq_length < len(l)):
        max_seq_length = len(l)
for l in agg_list_test_seq: # finding the max sequence length for each valid sequence
    if (max_seq_length < len(l)):
        max_seq_length = len(l)





# save
with open('../../data/pkl/varying_seq_length/train_v3.pkl', 'wb') as f:
    pickle.dump(agg_list_train_seq, f)

with open('../../data/pkl/varying_seq_length/valid_v3.pkl', 'wb') as f:
    pickle.dump(agg_list_valid_seq, f)

with open('../../data/pkl/varying_seq_length/test_v3.pkl', 'wb') as f:
    pickle.dump(agg_list_test_seq, f)




