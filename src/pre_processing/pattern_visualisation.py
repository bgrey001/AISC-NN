#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:25:15 2022

@author: BenedictGrey

Time series pre-processing for data visualisation for potential input image for a convolutional neural network (CNN)

NB: df.iloc[rows, columns]

"""

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import time
import calendar




# finding time range of 24 hours in the dataset and partitioning based on this

# function that takes a dataframe and a time period and returns a set of dataframes that correspond to the time series for an individual vessel in 24 hours
def segment_dataframe(dataframe, time_period, threshold):
    
    # list to be returned
    list_df = []
    
    # set first timestamp and the time difference threshold
    t0 = datetime.fromtimestamp(dataframe['timestamp'].iloc[0])
    delta_size = timedelta(hours=time_period)
    curr_dif = timedelta(hours=0)

    # temp make the df smaller for testing
    #dataframe = dataframe.head(sample_size)
    initial_index = 0

    for i in range(0, len(dataframe)):
        
        curr_t = datetime.fromtimestamp(dataframe['timestamp'].iloc[i])
        curr_dif = curr_t - t0 # difference from current time to first time
        # need to make sure time difference is >= time difference and that the ship's id is the same
        if (curr_dif >= delta_size and (dataframe.iloc[i]['mmsi'] == dataframe.iloc[initial_index]['mmsi'])):
            
            # make sure size is meaningful (greater than threshold)
            if ((i - initial_index) > threshold):     
                list_df.append(dataframe.iloc[initial_index : i, :])

                
            t0 = datetime.fromtimestamp(dataframe['timestamp'].iloc[i])
            initial_index = i

    return list_df


# function that creates dataframes for plotting - outputs a dataframe for each ship (from mmsi)
def plot_vessels(dataframe):
    vector = dataframe['mmsi'].unique()
    matrix_df = []
    for vessel in vector: 
        df = dataframe[dataframe['mmsi'] == vessel]
        matrix_df.append(df)
    return matrix_df

# method that takes a list of dataframes a number and bool rand. Plots a batch over a the specified number of time steps randomly or not
def random_vis(x_batches, n_plots, rand):
    for i in range(0, n_plots):
        if rand == True:
            r = random.randint(len(x_batches))
        else:
            r = i
        batch = x_batches[r]
        plt.plot(batch['lat'], batch['lon'])
        plt.show()








