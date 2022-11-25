#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:29:53 2022

@author: benedict

data processing test script
"""

import pandas as pd
import numpy as np


import pre_processing_module as ppm


module = ppm.pre_processing_module('trollers')

# interpolated
# list_train, list_valid, list_test = module.return_test_train_sets(uniform=False, interpolated=True)
x, y, z = module.return_test_train_sets(uniform=False, interpolated=True)

length = len(x[0])

# # uninterpolated, uniform
# list_train_2, list_valid_2, list_test_2 = module.return_test_train_sets(uniform=True)

# # uninterpolated, varying
# list_train_3, list_valid_3, list_test_3 = module.return_test_train_sets(uniform=False)



# module_2 = ppm.pre_processing_module('trawlers')

# # interpolated
# list_train_2, list_valid_2, list_test_2 = module_2.return_test_train_sets(uniform=False, interpolated=True)