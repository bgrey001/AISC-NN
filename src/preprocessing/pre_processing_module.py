#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 14:30:07 2022

@author: BenedictGrey

Pre-processing module class

This class has several helper methods and can be called to process a given dataset into a pytorch tensor ready for training

"""

import numpy as np
from numpy import random
import pandas as pd

import warnings
from pandas.core.common import SettingWithCopyWarning

import dask.dataframe as dd

import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning) # ignore warning for creating views instead of copies

class pre_processing_module:
    
    
    # =============================================================================
    # constructor method to initialise attributes, import dataset and execute integer encoding
    # =============================================================================
    def __init__(self, dataset):
        
        # init attributes
        self.dataset = dataset
        self.list_df = []
        self.list_df_resampled = []
        self.x_batches = []
        self.interpolated_vessels = []
        self.df = pd.read_csv('../../data/csv/' + dataset + '.csv')
        self.df['timestamp'] = pd.to_datetime(self.df.timestamp, unit='s')
        self.df = self.df.drop(columns=['source'])
        
        # fits the integer encoding model and transforms the label to it's given integer value
        le = LabelEncoder()
        strings = ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers']
        le.fit(strings)
        self.desired = dataset
        self.desired = le.transform([self.desired])
        self.df['desired'] = self.desired[0]
        
        self.list_train_data = []
        self.list_valid_data = []
        self.list_test_data = []
        

    # =============================================================================
    # helper method to divide the dataset based on their maritime mobile service identity (mmsi)
    # =============================================================================
    def partition_vessels(self):
        vessel_list = []
        df = self.df
        vessel_ids = df['mmsi'].unique()
        for v in vessel_ids:
            temp_df = df[df['mmsi'] == v].reset_index(drop=True)
            vessel_list.append(temp_df)
        return vessel_list
    
    
    # =============================================================================
    # helper method to break unique dataframes into time periods
    # =============================================================================
    def segment_dataframe(self, dataframe, time_period, threshold): # must be run on unique dataframes, output of partition_vessels will do nicely
        
        # list to be returned
        list_df = []
        # max_seq_length = 510
        
        # set first timestamp and the time difference threshold
        t0 = dataframe['timestamp'].iloc[0]
        delta_size = timedelta(hours=time_period)
        curr_dif = timedelta(hours=0)

        initial_index = 0
        for i in range(0, len(dataframe)):
            
            curr_t = dataframe['timestamp'].iloc[i]
            curr_dif = curr_t - t0 # difference from current time to first time

            if (curr_dif >= delta_size):
                # make sure size is meaningful (greater than threshold)
                if ((i - initial_index) > threshold):     
                    list_df.append(dataframe.iloc[initial_index : i, :].reset_index(drop=True))
                    
                # if uniform: # pad with zeroes to be the same length as the largest sequence
                t0 = dataframe['timestamp'].iloc[i]
                initial_index = i
                
                        
        return list_df
    
    
    # =============================================================================
    # helper method that featurises the time differences between items in the sequence
    # this should be used on already divided sequences
    # =============================================================================
    def create_deltas(self, list_df, interpolated): # also removing some undesirable columns
        total_num_seconds = 86400 # in 24 hours
        new_list = []
        for df in list_df:
            
            if not interpolated:
                # time deltas 
                df['delta_t'] = (df['timestamp']-df['timestamp'].shift()).fillna(pd.Timedelta(seconds=0)) # compute change in time for each timestep and create feature delta_t
                df['delta_t_cum'] = df['delta_t'].cumsum() # get cummalative sum of the delta column
                df['normalised_delta_t'] = df['delta_t'].dt.total_seconds() / total_num_seconds # normalising date time
            
            # course and speed deltas
            df['delta_c'] = (df['course']-df['course'].shift()).abs() # course difference
            df.loc[df['delta_c'] >= 180, 'delta_c'] -= 360 # in degrees so we need to make sure the range stays between 0 and 360
            df['delta_c'] = df['delta_c'].abs()
            # df['delta_s'] = (df['speed']-df['speed'].shift()).abs() # speed difference
            
            # convert lat and lon to x, y, z coordinates as they are 3D
            
            
            # df = df.drop(columns=drop_columns)
            df['tagets'] = df['desired'] # move targets to end
            df = df.drop(columns=['desired'])
            
            # remains to be seen if we need to scale lat and lon
            new_list.append(df)
        return new_list
    
    # =============================================================================
    # helper method that calls segment_dataframes
    # =============================================================================
    def build_sequences(self, list_df, time_period, threshold, max_seq_length, uniform, drop_columns=[]):

        df_list_seq = []
        normalised_list_seq = []
        
        # first, segment the datframes
        for df in list_df:    
            df_list_seq.append(self.segment_dataframe(dataframe=df, time_period=time_period, threshold=threshold))

            
        # # padding for creating uniform time sequences
        # for i, df_list in enumerate(df_list_seq):
        #     # print(i)
        #     for j, df in enumerate(df_list):
        #         if uniform:
        #             df_list[j] = df_list[j].reindex(list(range(0, max_seq_length))).reset_index(drop=True) # padding operation
        #         df_list[j] = df_list[j].reset_index(drop=True) # reset indices
        #         df_list[j] = df_list[j].drop(columns=drop_columns) # drop columns
        
        # return df_list_seq
        sequences = [item for sub_list in df_list_seq for item in sub_list]
        
        # convert to numpy arrays
        for i in range(len(sequences)): 
            sequences[i] = sequences[i].to_numpy()
            
        # print(f'Number of sequences in the dataset: {len(sequences)}')
        return sequences

    # =============================================================================
    # helper method that calls build_sequences on the train and test lists to return the fully processed lists of sequences
    # =============================================================================
    def return_test_train_sets(self, uniform, interpolated=False):
        
        if interpolated:
            time_interval = 5 # in minutes, how far apart each interval should be
            return self.return_interpolated_data(time_interval=time_interval, fixed_window=True)
       
        drop_columns = ['mmsi', 'timestamp', 'distance_from_shore', 'distance_from_port', 'delta_t', 'delta_t_cum', 'course', 'speed']
        
        max_seq_length = 1887 # this should be set the largest sequence length in all the datasets
        threshold = 150 # this is the minimum that a sequence can be to be included in the dataset
        time_period = 24 # this is the length of each time window in hours
        
        
        df = self.partition_vessels()
        df_list = self.create_deltas(list_df=df, interpolated=False)
        list_train, list_valid, list_test = self.split(df_list)

        list_train_seq = self.build_sequences(list_df=list_train, time_period=time_period, threshold=threshold, max_seq_length=max_seq_length, uniform=uniform, drop_columns=drop_columns)
        list_valid_seq = self.build_sequences(list_df=list_valid, time_period=time_period, threshold=threshold, max_seq_length=max_seq_length, uniform=uniform, drop_columns=drop_columns)
        list_test_seq = self.build_sequences(list_df=list_test, time_period=time_period, threshold=threshold, max_seq_length=max_seq_length, uniform=uniform, drop_columns=drop_columns)

        print(f'Number of train sequences in the dataset: {len(list_train_seq)}')
        print(f'Number of val sequences in the dataset: {len(list_valid_seq)}')
        print(f'Number of test sequences in the dataset: {len(list_test_seq)}')
        
        return list_train_seq, list_valid_seq, list_test_seq
        
    

    # =============================================================================
    # helper method that splits the data into train and test sets and then normalises based on the data in the train sets
    # =============================================================================
    def split(self, list_df): # takes as input the output of create_deltas()
    
        list_train = []
        list_valid = []
        list_test = []
        
        # merge dataframes
        df = pd.concat(list_df)
        
        # now split
        dfx = df.iloc[:, :-1]
        dfy = df.iloc[:, -1]
        # shuffle must be false as we need to preserve the order of the time sequences 
        X_train, X_rem, y_train, y_rem = train_test_split(dfx, dfy, train_size=0.8, random_state=None, shuffle=False, stratify=None)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=None, shuffle=False, stratify=None)

        # now reassign the targets to the datasets for transporting
        df_train = X_train
        df_train['desired'] = y_train
        
        df_valid = X_valid
        df_valid['desired'] = y_valid

        df_test = X_test
        df_test['desired'] = y_test
        list_train.append(df_train)
        list_valid.append(df_valid)
        list_test.append(df_test)

        
        return list_train, list_valid, list_test


    
    
    # @classmethod
    # =============================================================================
    # @output returns a list of np.arrays containing the interpolated values for lat and long (currently - planning to expand to have other features)   
    # @params - vessel list is output of partition_vessels() method, target interval is the size of the time steps
    # =============================================================================
    def linear_interpolation(self, vessel_list, target_interval):
        
        list_output_resampled = []
        list_output = []
    
        for v in vessel_list:
            
            v.index = v['timestamp']
            v.index.name = 'index'

            # invert boolean series to remove duplicates of timestamps
            v = v[~v.index.duplicated(keep='first')]
            list_output.append(v)

            # upsample the dataframe using the mean dispatching function
            v = v.resample(str(target_interval) + 'T').mean()
            
            # use linear interpolation to fill the gaps in the data
            v = v.interpolate(method='linear')
     
            # list_output_resampled.append(np.array(v.values)) # coverts to numpy array
            list_output_resampled.append(v)
            # self.list_df_resampled.append(v) # returns dataframe
            
        
        #total_average = total_average/5 # use to see average of time steps in the dataset
        # self.interpolated_vessels = list_output_resampled
        # self.list_df = list_output
        return list_output_resampled




    # @classmethod
    # =============================================================================
    # segment into 24 hour windows by taking t_steps timesteps and putting them in a batch sequentially (all batches are spaced by t_steps)
    # @outputs a tensor of input values and a tensor of target values
    # =============================================================================
    def fixed_window(self, interpolated_list, t_steps):
        
        batches = []
        t_steps = int(t_steps)
        drop_columns = ['mmsi', 'distance_from_shore', 'distance_from_port', 'is_fishing', 'desired']

        
        for i, df in enumerate(interpolated_list):
            # df['tagets'] = df['desired'] # move targets to end
            # print(interpolated_list[i]['desired'])
            interpolated_list[i]['target'] = interpolated_list[i]['desired']
            interpolated_list[i] = interpolated_list[i].drop(columns=drop_columns) # drop columns
            interpolated_list[i] = np.array(interpolated_list[i].values) # convert to numpy array
        
        for v in interpolated_list:
            for i in range(0, len(v), t_steps):
                # if the end of the vessel sequence isn't long enough we need to take some other values
                if len(v[i:(i+t_steps), :]) < t_steps:
                    length_batch = len(v[i:(i+t_steps), :])
                    diff = t_steps - length_batch
                    batches.append(v[i-diff:(i+t_steps), :])
                else:    
                    batches.append(v[i:(i+t_steps), :])
        return batches


    # @classmethod
    # =============================================================================
    # segment into 24 hour windows that start one hour after each other incrementally ** more expesive than fixed_window_tensor
    # @outputs a tensor of input values and a tensor of target values
    # =============================================================================
    def sliding_window(self, interpolated_list, t_steps, lag):
        batches = []
        t_steps = int(t_steps)
        drop_columns = ['mmsi', 'distance_from_shore', 'distance_from_port', 'course', 'is_fishing']
        
        for i, df in enumerate(interpolated_list):

            interpolated_list[i] = interpolated_list[i].drop(columns=drop_columns) # drop columns
            interpolated_list[i] = np.array(interpolated_list[i].values) # convert to numpy array
        
        for v in interpolated_list:
            for i in range(0, len(v), lag):
                # if the end of the vessel sequence isn't long enough we need to take some other values
                if len(v[i:(i+t_steps), :]) < t_steps:
                    length_batch = len(v[i:(i+t_steps), :])
                    diff = t_steps - length_batch
                    batches.append(v[i-diff:(i+t_steps), :])
                else:    
                    batches.append(v[i:(i+t_steps), :])
        return batches
      
    
    
    # =============================================================================
    # method to remove outliers with insignificant course changes
    # =============================================================================
    def remove_outliers(self, batches): # takes as input the output of the sliding_window functions
        
        
        
        
        return
        
        
        
        
    # =============================================================================
    # method to return the linearly interpolated time series data in train, test and valid splits
    # =============================================================================
    def return_interpolated_data(self, time_interval, fixed_window):
        
        # variable dependant on what's required to make 24 hour batches
        batch_size = 1440 / time_interval
        lag = 3 #  lag to delay the sliding window
        
        vessel_list = self.partition_vessels()
        interpolated_vessels = self.linear_interpolation(vessel_list=vessel_list, target_interval=time_interval)
        
        
        # remove outliers step
        
        
        
        # df_list = self.create_deltas(list_df=interpolated_vessels, interpolated=True)
        
        # return self.split(df_list) # temp
        
        
        if fixed_window:
            df_list = self.fixed_window(interpolated_vessels, batch_size)
        else:
            df_list = self.sliding_window(interpolated_vessels, batch_size, lag)
            
        return df_list
    
        # print(f'Number of train sequences in the dataset: {len(self.list_train_data)}')
        # print(f'Number of val sequences in the dataset: {len(self.list_valid_data)}')
        # print(f'Number of test sequences in the dataset: {len(self.list_test_data)}')
        
        
        # list_train, list_valid, list_test = self.split(df_list)

        # return self.list_train_data, self.list_valid_data, self.list_test_data
    
    
    
    # # =============================================================================
    # # method to return the linearly interpolated time series data in train, test and valid splits
    # # =============================================================================
    # def return_interpolated_data(self, time_interval, fixed_window):
        
    #     # variable dependant on what's required to make 24 hour batches
    #     batch_size = 1440 / time_interval
    #     lag = 3 #  lag to delay the sliding window
        
    #     vessel_list = self.partition_vessels()
    #     interpolated_vessels = self.linear_interpolation(vessel_list=vessel_list, target_interval=time_interval)
        
        
    #     # remove outliers step
        
        
        
    #     df_list = self.create_deltas(list_df=interpolated_vessels, interpolated=True)
        
    #     # return self.split(df_list) # temp
        
    #     list_train, list_valid, list_test = self.split(df_list)
        
    #     if fixed_window:
    #         self.list_train_data = self.fixed_window(list_train, batch_size)
    #         self.list_valid_data = self.fixed_window(list_valid, batch_size)
    #         self.list_test_data = self.fixed_window(list_test, batch_size)
    #     else:
    #         self.list_train_data = self.sliding_window(list_train, batch_size, lag)
    #         self.list_valid_data = self.sliding_window(list_valid, batch_size, lag)
    #         self.list_test_data = self.sliding_window(list_test, batch_size, lag)
            
        
    #     print(f'Number of train sequences in the dataset: {len(self.list_train_data)}')
    #     print(f'Number of val sequences in the dataset: {len(self.list_valid_data)}')
    #     print(f'Number of test sequences in the dataset: {len(self.list_test_data)}')
            
    #     return self.list_train_data, self.list_valid_data, self.list_test_data
            
    
    # @classmethod
    # =============================================================================
    # method that automates the instantiation and building of the tensor 
    # @param time_interval: is in minues - how long the time windows will be
    # @param sliding_window: bool value that creates a sliding window if True and uses fixed windows of 24 hours if not
    # =============================================================================
    def build_tensor(self, time_interval, sliding_window):
        
        # variable dependant on what's required to make 24 hour batches
        batch_size = 1440 / time_interval
        
        ppm = pre_processing_module(self.dataset)
        vessel_list = ppm.partition_vessels()
        iv = ppm.linear_interpolation(vessel_list, time_interval)
        
        if (sliding_window == True):
            return ppm.sliding_window_tensor(iv, batch_size)
        else:
            return ppm.fixed_window_tensor(iv, batch_size)
    
    
    # @classmethod
    # =============================================================================
    # helper method for data vis
    # =============================================================================
    def vis_colour(self, batch, speed_min, speed_max, marzuki):
        # anything under or over speed_range will be plotted in black, otherwise in red 
        
        df_vis = pd.DataFrame(batch)
        
        if marzuki == True:
            # marzuki speed categorisation
            df_vis_red = df_vis[df_vis[5].between(speed_min, speed_max, inclusive=True)]
            df_vis_black = df_vis[df_vis[5].between(speed_min, speed_max, inclusive=True) != True]
        else:
            # if not zero
            df_vis_red = df_vis[df_vis[5] > 0]
            df_vis_black = df_vis[df_vis[5] == 0]
        
        plt.plot(df_vis_red[1], df_vis_red[2], c='red')
        plt.plot(df_vis_black[1], df_vis_black[2], c='black')
        
    # =============================================================================
    # helper method for data vis
    # =============================================================================
    def vis(self, batch):
        df_vis = pd.DataFrame(batch)
        plt.plot(df_vis[1], df_vis[2])
        
    # =============================================================================
    # helper method for data vis
    # =============================================================================   
    def visualise_batch(self, x_batches, n_plots, rand):
        for i in range(0, n_plots):
            if rand == True:
                r = random.randint(len(x_batches))
            else:
                r = i
            batch = x_batches[r]
            self.vis(batch)
            plt.show()

        
    # @classmethod   
    # =============================================================================
    # method for examaning data upsampling ratios
    # =============================================================================
    def error_checking(self):
        size = 0
        for v in self.interpolated_vessels:
            size = size + len(v)
        size_batches = 96 * self.x_tensor.shape[0]
        print("size of all interpolated values:", size, "size of aggregate batches:", size_batches)
        
        
    # @classmethod
    def return_size(self):
        return len(self.build_sequences())




# =============================================================================
# ------------------------------------------------------------------------------ # ------------------------------------------------------------------------------
# end class
# ------------------------------------------------------------------------------ # ------------------------------------------------------------------------------
# =============================================================================



# =============================================================================
# testing zone
# =============================================================================



