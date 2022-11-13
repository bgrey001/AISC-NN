#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:14:23 2022

@author: BenedictGrey

Script to calculate a feature variable of the time difference as an alternative to linear interpolation
"""

import pandas as pd
import numpy as np

# import dataset and process to make timestamp index
df = pd.read_csv('../data/purse_seines.csv').head(15)
df['timestamp'] = pd.to_datetime(df.timestamp, unit='s')
df = df.drop(columns=['source'])
df.index = df['timestamp']
df = df[~df.index.duplicated(keep='first')]


# compute change in time for each timestep and create feature delta_t
df['delta_t'] = (df['timestamp']-df['timestamp'].shift()).fillna(pd.Timedelta(seconds=0))
df = df.drop(columns=['timestamp'])
# get cummalative sum of the delta column
df['delta_t'] = df['delta_t'].cumsum()






