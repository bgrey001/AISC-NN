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
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

class pre_processing_module:
    
    @classmethod
# =============================================================================
#     constructor method to initialise attributes, import dataset and execute integer encoding
# =============================================================================
    def __init__(self, dataset):
        # init attributes
        self.list_df = []
        self.list_df_resampled = []
        self.x_tensor = torch.empty(0,0)
        self.x_batches = []
        self.dataset = dataset
        self.desired = dataset
        self.interpolated_vessels = []
        self.df = pd.read_csv('../data/' + dataset + '.csv')
        # convert to readable timestamp
        self.df['timestamp'] = pd.to_datetime(self.df.timestamp, unit='s')
        self.df = self.df.drop(columns=['source'])
        
        # fits the integer encoding model and transforms the label to it's given integer value
        le = LabelEncoder()
        strings = ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers']
        le.fit(strings)
        self.desired = le.transform([self.desired])
        self.df['desired'] = self.desired[0]
        
        
    @classmethod
# =============================================================================
#     helper method to divide the dataset based on their maritime mobile service identity (mmsi)
# =============================================================================
    def create_set(self):
        vessel_list = []
        vessel_ids = self.df['mmsi'].unique()
        for v in vessel_ids:
            temp_df = self.df[self.df['mmsi'] == v]
            vessel_list.append(temp_df)
        return vessel_list
    
    
    @classmethod
# =============================================================================
#     @output returns a list of np.arrays containing the interpolated values for lat and long (currently - planning to expand to have other features)   
#     @params - vessel list is output of create_set() method, target interval is the size of the time steps
# =============================================================================
    def interpolate_vessels(self, vessel_list, target_interval):
        
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
#     segment into 24 hour windows by taking t_steps timesteps and putting them in a batch sequentially (all batches are spaced by t_steps)
#     @outputs a tensor of input values and a tensor of target values
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
        self.x_tensor = torch.tensor(x_batches)
        self.x_batches = x_batches
        return torch.tensor(x_batches), torch.tensor(y_batches)



    @classmethod
# =============================================================================
#     segment into 24 hour windows that start one hour after each other incrementally ** more expesive than fixed_window_tensor
#     @outputs a tensor of input values and a tensor of target values
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
        self.x_tensor = torch.tensor(x_batches)
        self.x_batches = x_batches
        #return torch.tensor(x_batches), x_batches
        return torch.tensor(x_batches), torch.tensor(y_batches)
        #return torch.tensor(x_batches), x_batches, torch.tensor(y_batches), y_batches
    
    
    @classmethod
# =============================================================================
#     method that automates the instantiation and building of the tensor 
#     @param time_interval: is in minues - how long the time windows will be
#     @param sliding_window: bool value that creates a sliding window if True and uses fixed windows of 24 hours if not
# =============================================================================
    def build_tensor(self, time_interval, sliding_window):
        
        # variable dependant on what's required to make 24 hour batches
        batch_size = 1440 / time_interval
        
        ppm = pre_processing_module(self.dataset)
        vessel_list = ppm.create_set()
        iv = ppm.interpolate_vessels(vessel_list, time_interval)
        
        if (sliding_window == True):
            return ppm.sliding_window_tensor(iv, batch_size)
        else:
            return ppm.fixed_window_tensor(iv, batch_size)
    
    
    @classmethod
# =============================================================================
#     helper method to 
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
        

    def vis(self, batch):
        df_vis = pd.DataFrame(batch)
        plt.plot(df_vis[1], df_vis[2])
        
        
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
#     method for examaning data upsampling ratios
# =============================================================================
    def error_checking(self):
        size = 0
        for v in self.interpolated_vessels:
            size = size + len(v)
        size_batches = 96 * self.x_tensor.shape[0]
        print("size of all interpolated values:", size, "size of aggregate batches:", size_batches)
        
        
    @classmethod
    def return_dataframe(self):
        return self.list_df_resampled





# =============================================================================
# ------------------------------------------------------------------------------ # ------------------------------------------------------------------------------
# end class
# ------------------------------------------------------------------------------ # ------------------------------------------------------------------------------
# =============================================================================



# =============================================================================
# testing zone
# =============================================================================

ppm = pre_processing_module('purse_seines')
x_tensor, y_tensor = ppm.build_tensor(time_interval=15, sliding_window=False)

threshold = 20000
t2 = 5000

array_np = ppm.interpolated_vessels[0][0:threshold, :]
# =============================================================================
# plt.scatter(array_np[:,5], array_np[:,6], s=1, c='black')
# =============================================================================
plt.plot(array_np[:,5], array_np[:,6])
plt.show()

df1 = ppm.list_df_resampled[0].head(threshold)
# =============================================================================
# plt.scatter(df1['lat'], df1['lon'], s=1, c='black')
# =============================================================================
plt.plot(df1['lat'], df1['lon'], c='green')
plt.show()


df2 = ppm.list_df[0].head(t2)
df2_1 = df2.resample('15T').bfill()
df2_2 = df2.resample('15T').mean()
df2_2 = df2_2.interpolate(method='linear')
# =============================================================================
# plt.scatter(df2_1['lat'], df2_1['lon'], s=1, c='black')
# =============================================================================
plt.plot(df2_1['lat'], df2_1['lon'], c='green')
plt.plot(df2_2['lat'], df2_2['lon'], c='brown')
plt.show()

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

