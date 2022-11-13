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

import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning) # ignore warning for creating views instead of copies

class pre_processing_module:
    
    @classmethod
    # =============================================================================
    # constructor method to initialise attributes, import dataset and execute integer encoding
    # =============================================================================
    def __init__(self, dataset):
        
        # init attributes
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
        
        
    @classmethod
    # =============================================================================
    # helper method to divide the dataset based on their maritime mobile service identity (mmsi)
    # =============================================================================
    def partition_vessels(self):
        vessel_list = []
        vessel_ids = self.df['mmsi'].unique()
        for v in vessel_ids:
            temp_df = self.df[self.df['mmsi'] == v]
            vessel_list.append(temp_df)
        return vessel_list
    
    
    @classmethod
    # =============================================================================
    # helper method to break unique dataframes into time periods
    # =============================================================================
    def segment_dataframe(self, dataframe, time_period, threshold): # must be run on unique dataframes, output of partition_vessels will do nicely
        
        # list to be returned
        list_df = []
        
        # set first timestamp and the time difference threshold
        t0 = dataframe['timestamp'].iloc[0]
        delta_size = timedelta(hours=time_period)
        curr_dif = timedelta(hours=0)

        # temp make the df smaller for testing
        # dataframe = dataframe.head(50)
        initial_index = 0

        for i in range(0, len(dataframe)):
            
            curr_t = dataframe['timestamp'].iloc[i]
            curr_dif = curr_t - t0 # difference from current time to first time

            if (curr_dif >= delta_size):
                # make sure size is meaningful (greater than threshold)
                if ((i - initial_index) > threshold):     
                    list_df.append(dataframe.iloc[initial_index : i, :])
                t0 = dataframe['timestamp'].iloc[i]
                initial_index = i

        return list_df
    
    
    @classmethod
    # =============================================================================
    # helper method that featurises the time differences between items in the sequence
    # =============================================================================
    def create_deltas(self, list_df): # also removing some undesirable columns, should refactor
        total_num_seconds = 86400
        new_list = []
        for df in list_df:
            # compute change in time for each timestep and create feature delta_t
            df['delta_t'] = (df['timestamp']-df['timestamp'].shift()).fillna(pd.Timedelta(seconds=0))
            # get cummalative sum of the delta column
            df['delta_t_cum'] = df['delta_t'].cumsum()
            # print(type(df['delta_t']))
            df['normalised_delta'] = df['delta_t'].dt.total_seconds() / total_num_seconds # normalising date time
            df = df.drop(columns=['mmsi', 'timestamp', 'distance_from_shore', 'distance_from_port', 'is_fishing', 'delta_t', 'delta_t_cum'])
            # move targets to end
            df['target'] = df['desired']
            df = df.drop(columns=['desired'])
            
            # remains to be seen if we need to scale lat and lon
        
            new_list.append(df)
        return new_list
    
    @classmethod
    # =============================================================================
    # helper method that calls segment_dataframes and create_deltas to form the final sequences for feeding into the training algorithms
    # =============================================================================
    def build_sequences(self, list_df, time_period, threshold):
        # list_df = self.partition_vessels() # need to init list_df
        df_list_seq = []
        normalised_list_seq = []
        # first, segment the datframes
        for df in list_df:    
            df_list_seq.append(self.segment_dataframe(dataframe=df, time_period=time_period, threshold=threshold))
        
        for sub_list in df_list_seq:
            normalised_list_seq.append(self.create_deltas(sub_list))
            
        sequences = [item for sub_list in normalised_list_seq for item in sub_list]
        # convert to numpy arrays
        for i in range(len(sequences)): 
            sequences[i] = sequences[i].to_numpy()
            
        # print(f'Number of sequences in the dataset: {len(sequences)}')
            
        return sequences

    @classmethod
    # =============================================================================
    # helper method that calls build_sequences on the train and test lists to return the fully processed lists of sequences
    # =============================================================================
    def return_test_train_sets(self, train_max_length, valid_max_length, test_max_length):
        df = self.partition_vessels()
        list_train, list_valid, list_test = self.split_scale(df)
        list_train_seq = self.build_sequences(list_df=list_train, time_period=24, threshold=4)
        list_valid_seq = self.build_sequences(list_df=list_valid, time_period=24, threshold=4)
        list_test_seq = self.build_sequences(list_df=list_test, time_period=24, threshold=4)
        
        print(f'Number of sequences in the dataset: {len(list_train_seq)}')
        
        return list_train_seq[:train_max_length], list_valid_seq[:valid_max_length], list_test_seq[:test_max_length]

        
    

    @classmethod
    # =============================================================================
    # helper method that splits the data into train and test sets and then normalises based on the data in the train sets
    # =============================================================================
    def split_scale(self, list_df): # takes as input the output of partition_vessels()
        list_train = []
        list_valid = []
        list_test = []
        for df in list_df:
            
            dfx = df.iloc[:, :-1]
            dfy = df.iloc[:, -1]
            # shuffle must be false as we need to preserve the order of the time sequences 
            X_train, X_rem, y_train, y_rem = train_test_split(dfx, dfy, train_size=0.8, random_state=None, shuffle=False, stratify=None)
            X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=None, shuffle=False, stratify=None)
            
            scaler = MinMaxScaler()
            X_train[['speed', 'course', 'lat', 'lon']] = scaler.fit_transform(X_train[['speed', 'course', 'lat', 'lon']])
            X_valid[['speed', 'course', 'lat', 'lon']] = scaler.transform(X_valid[['speed', 'course', 'lat', 'lon']])
            X_test[['speed', 'course', 'lat', 'lon']] = scaler.transform(X_test[['speed', 'course', 'lat', 'lon']])
            
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
            



    @classmethod
    # =============================================================================
    # helper method that normalises the features of an np array of sequences returned by build_sequences
    # =============================================================================
    def normalise_sequences(self, sequences):
        for i in range(len(sequences)):
            scaler = MinMaxScaler()
            features = sequences[i][:, :-1]
            sequences[i][:, :-1] = scaler.fit_transform(features)
        return sequences

    
    
    @classmethod
    # =============================================================================
    # @output returns a list of np.arrays containing the interpolated values for lat and long (currently - planning to expand to have other features)   
    # @params - vessel list is output of partition_vessels() method, target interval is the size of the time steps
    # =============================================================================
    def linear_interpolation(self, vessel_list, target_interval):
        
        list_output_resampled = []
        list_output = []
    
        for v in vessel_list:
        
            v.index = v['timestamp']
            v = v.drop(columns=['timestamp'])

            # invert boolean series to remove duplicates of timestamps
            v = v[~v.index.duplicated(keep='first')]
            list_output.append(v)

            # upsample the dataframe using the mean dispatching function
            v = v.resample(str(target_interval) + 'T').mean()
            
            # use linear interpolation to fill the gaps in the data
            v = v.interpolate(method='linear')
     
            list_output_resampled.append(np.array(v.values)) # proper return value
            self.list_df_resampled.append(v) # returns dataframe
            
        
        #total_average = total_average/5 # use to see average of time steps in the dataset
        self.interpolated_vessels = list_output_resampled
        self.list_df = list_output
        return list_output_resampled

        
    @classmethod
    # =============================================================================
    # segment into 24 hour windows by taking t_steps timesteps and putting them in a batch sequentially (all batches are spaced by t_steps)
    # @outputs a tensor of input values and a tensor of target values
    # =============================================================================
    def fixed_window_tensor(self, interpolated_vessels, t_steps):
        x_batches = []
        y_batches = []
        t_steps = int(t_steps)
        for v in interpolated_vessels:
            for i in range(0, len(v), t_steps):
                # if the end of the vessel sequence isn't long enough we need to take some other values
                if len(v[i:(i+t_steps), :]) < t_steps:
                    length_batch = len(v[i:(i+t_steps), :])
                    diff = t_steps - length_batch
                    x_batches.append(v[i-diff:(i+t_steps), :-1])
                    y_batches.append(v[i-diff:(i+t_steps), -1])
                else:    
                    x_batches.append(v[i:(i+t_steps), :-1])
                    y_batches.append(v[i:(i+t_steps), -1])
        #return torch.tensor(batches), batches
        # self.x_tensor = torch.tensor(x_batches)
        self.x_batches = x_batches
        self.y_batches = y_batches
        return torch.tensor(x_batches), torch.tensor(y_batches)



    @classmethod
    # =============================================================================
    # segment into 24 hour windows that start one hour after each other incrementally ** more expesive than fixed_window_tensor
    # @outputs a tensor of input values and a tensor of target values
    # =============================================================================
    def sliding_window_tensor(self, interpolated_vessels, t_steps, lag):
        x_batches = []
        y_batches = []
        for v in interpolated_vessels:
            for i in range(0, len(v), lag):
                if len(v[i:(i+t_steps), :]) < t_steps:
                    length_batch = len(v[i:(i+t_steps), :])
                    diff = t_steps - length_batch
                    x_batches.append(v[i-diff:(i+t_steps), :])
                    y_batches.append(v[i-diff:(i+t_steps), -1])
                    
                else:    
                    x_batches.append(v[i:(i+t_steps), :])
                    y_batches.append(v[i:(i+t_steps), -1])
        # self.x_tensor = torch.tensor(x_batches)
        self.x_batches = x_batches
        self.y_batches = y_batches
        #return torch.tensor(x_batches), x_batches
        return torch.tensor(x_batches), torch.tensor(y_batches)
        #return torch.tensor(x_batches), x_batches, torch.tensor(y_batches), y_batches
    
    
    @classmethod
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
    
    
    @classmethod
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

        
    @classmethod   
    # =============================================================================
    # method for examaning data upsampling ratios
    # =============================================================================
    def error_checking(self):
        size = 0
        for v in self.interpolated_vessels:
            size = size + len(v)
        size_batches = 96 * self.x_tensor.shape[0]
        print("size of all interpolated values:", size, "size of aggregate batches:", size_batches)
        
        
    @classmethod
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
# =============================================================================
# 
# ppm = pre_processing_module('purse_seines')
# x_tensor, y_tensor = ppm.build_tensor(time_interval=15, sliding_window=False)
# 
# threshold = 20000
# t2 = 5000
# 
# array_np = ppm.interpolated_vessels[0][0:threshold, :]
# # =============================================================================
# # plt.scatter(array_np[:,5], array_np[:,6], s=1, c='black')
# # =============================================================================
# plt.plot(array_np[:,5], array_np[:,6])
# plt.show()
# 
# df1 = ppm.list_df_resampled[0].head(threshold)
# # =============================================================================
# # plt.scatter(df1['lat'], df1['lon'], s=1, c='black')
# # =============================================================================
# plt.plot(df1['lat'], df1['lon'], c='green')
# plt.show()
# 
# 
# df2 = ppm.list_df[0].head(t2)
# df2_1 = df2.resample('15T').bfill()
# df2_2 = df2.resample('15T').mean()
# df2_2 = df2_2.interpolate(method='linear')
# # =============================================================================
# # plt.scatter(df2_1['lat'], df2_1['lon'], s=1, c='black')
# # =============================================================================
# plt.plot(df2_1['lat'], df2_1['lon'], c='green')
# plt.plot(df2_2['lat'], df2_2['lon'], c='brown')
# plt.show()
# 
# =============================================================================
# =============================================================================
# # original
# plt.scatter(df2['lat'], df2['lon'], s=15)
# plt.plot(df2['lat'], df2['lon'])
# 
# # resampled but not interpolated
# # =============================================================================
# # plt.scatter(df2_1['lat'], df2_1['lon'], s=15)
# # =============================================================================
# plt.plot(df2_1['lat'], df2_1['lon'])
# =============================================================================

# =============================================================================
# x_tensor, x_batches, y_tensor, y_batches = ppm.build_tensor(time_interval=15, sliding_window=False)
# ppm.error_checking()
# =============================================================================

# =============================================================================
# df_list = ppm.return_dataframe()
# df_test = ppm.df
# print("length of df_list:",len(df_list), "unique mmsi's:", len(df_test['mmsi'].unique()))
# df = df_list[0]
# =============================================================================

# =============================================================================
# len(x_batches)
# =============================================================================

# =============================================================================
# for i in range(0, 6369):
#     r = random.randint(len(x_batches))
#     batch = x_batches[i]
#     #ppm.vis_colour(batch, 2, 6, False)
#     ppm.vis(batch)
# =============================================================================

# =============================================================================
# save as tensor file for deploying on the network
# =============================================================================
#torch.save(tensor, 'objects/trollers_tensor.pt')

