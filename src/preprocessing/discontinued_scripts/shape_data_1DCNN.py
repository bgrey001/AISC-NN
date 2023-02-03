#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:52:03 2022

@author: BenedictGrey

Script to segment data into time windows


Steps:
    1) Divide the dataset based on mmsi
    2) Interpolate values for each sub dataframe so they can be divided into multiples of 5 say
    3) Create the batches of 24hrs for each vessel
    4) Populate a 3D tensor with the data including each time window (sliding window)


"""
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import time
import calendar

# test dataset using trollers as its small
df = pd.read_csv('data/trollers.csv').tail(20)

# convert to readable timestamp
#df['timestamp'] = pd.to_datetime(df.timestamp, unit='s')


# first we need regular time series data, that will be done in a seperate scrip
# for the purpose of this script we will assume the time series is regular


# step 1, divide the dataset based on mmsi
def create_set(dataframe):
    vector = dataframe['mmsi'].unique()
    return vector


# reassign df to a single vessel sub dataframe
vector = create_set(df)
df = df[df['mmsi'] == vector[0]]

"""
Measuring skewness of distribution of the timestamps in the data
Skewness measures the asymmetry in the data
"""


df_time = df['timestamp']
df_time.skew()



print(df)










def linear_interpolate_set(sub_dataframe):
    



def create_segments(dataframe, segment_size):
    current_index = 0
    batches = []
    if (dataframe.iloc[current_index]['mmsi'] == dataframe.iloc[segment_size]['mmsi']):
        batches.append(dataframe.iloc[current_index:segment_size])
    

    
    
    
    
    
    
    
    
    
    