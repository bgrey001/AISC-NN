#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:20:12 2022

@author: BenedictGrey

Calculating common sense baseline:
    
    Training sizes:
        trollers = 1742
        pole and line = 2498
        purse seines = 20113
        fixed gear = 19420
        trawlers = 39236
        drifting longlines = 113578
        
        Aggregate = 196587
        
        agg = 1742 + 2498 + 20113 + 19420 + 39236 + 113578
        print(agg)
        csb = 113578/agg * 100

"""

import pandas as pd
import numpy as np
import pickle
import torch

import pre_processing_module as ppm

train_max_length = 1742
valid_max_length = 220
test_max_length = 266



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





