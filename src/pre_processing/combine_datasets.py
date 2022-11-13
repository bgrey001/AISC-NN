#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 01:12:47 2022

@author: benedict

Script to process the data using the pre_processing_module into train and test sets for all the datasets and combine them into 3 large sets:
training validation and testing data and then export using pickle ready for loading in the network scripts


"""

import pandas as pd
import numpy as np
import pickle
import torch

import pre_processing_module as ppm

"""
IMPORTANT: Need to set the number of sequences to be even across the classes so they don't introduce bias into the trianing algorithms

Number of training sequences for two smallest datasets:
    pole_and_line = 2192
    trollers = 1700
    
Number of validation sequences for two smallest dataest:
    pole_and_line = 228
    trollers = 217
    
Number of test sequences for two smallest dataest:
    pole_and_line = 240
    trollers = 252
        
Minimum value chosen for each

"""
train_max_length = 1700
valid_max_length = 228
test_max_length = 240

# =============================================================================
# trolers
# =============================================================================
trollers_module = ppm.pre_processing_module('trollers')
trollers_list_train_seq, trollers_list_valid_seq, trollers_list_test_seq = trollers_module.return_test_train_sets(train_max_length, valid_max_length, test_max_length)


# =============================================================================
# pole and line
# =============================================================================
pole_and_line_module = ppm.pre_processing_module('pole_and_line')
pole_and_line_list_train_seq, pole_and_line_list_valid_seq, pole_and_line_list_test_seq = pole_and_line_module.return_test_train_sets(train_max_length, valid_max_length, test_max_length)

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

# save
with open('../../data/pkl/train.pkl', 'wb') as f:
    pickle.dump(agg_list_train_seq, f)

with open('../../data/pkl/valid.pkl', 'wb') as f:
    pickle.dump(agg_list_valid_seq, f)

with open('../../data/pkl/test.pkl', 'wb') as f:
    pickle.dump(agg_list_test_seq, f)




