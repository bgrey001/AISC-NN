#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:04:38 2022

@author: BenedictGrey

This program is for simply generating a dataset to refer to the one hot encoding for each individual vessel dataset

The csv will be referred to during the data preprocessing phase. 
"""

import pandas as pd
import numpy as np

cols = ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers']

df = pd.DataFrame(columns=cols)

# generate matrix for one hot encoding
matrix = [];
for i in range(0, len(cols)):
    array = []
    for j in range(0, len(cols)):
        if j == i:
            array.append(1)
        else:
            array.append(0)
    matrix.append(array)
    

# populate dataframe with the matrix
for i in range(0, len(cols)):
    df.loc[i] = matrix[i]


# export to csv
df.to_csv('one_hot_encoded_classes.csv', index=False)
