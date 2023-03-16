#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Nov 24 14:30:07 2022
@author: Benedict Grey

Pre-processing module class current version

"""
# =============================================================================
# dependencies
# =============================================================================
import numpy as np
from numpy import random
import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning) # ignore warning for creating views instead of copies


# =============================================================================
# all purpose preprocessing utility used in multiple scripts across AISC-NN
# =============================================================================
class pre_processing_module:
    # =============================================================================
    # constructor method to initialise attributes, import dataset and execute integer encoding
    # =============================================================================
    def __init__(self, dataset, sample_size=None):
        if dataset:
            print(f'Processing {dataset} dataset...')
            self.dataset = dataset
            self.df = pd.read_csv('../../data/csv/' + dataset + '.csv')
            self.df['timestamp'] = pd.to_datetime(self.df.timestamp, unit='s')
            self.df = self.df.drop(columns=['source'])
            
            if sample_size is not None:
                self.df = self.df.head(sample_size)
            
            # fits the integer encoding model and transforms the label to it's given integer value
            le = LabelEncoder()
            strings = ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers']
            le.fit(strings)
            self.desired = dataset
            self.desired = le.transform([self.desired])
            self.df['desired'] = self.desired[0]

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
    def create_deltas(self, list_df, interpolated, drop_columns): # also removing some undesirable columns
        total_num_seconds = 86400 # in 24 hours
        for df in list_df:
            if not interpolated:
                # time deltas 
                df['delta_t'] = (df['timestamp']-df['timestamp'].shift()).fillna(pd.Timedelta(seconds=0)) # compute change in time for each timestep and create feature delta_t
                df['delta_t_cum'] = df['delta_t'].dt.total_seconds().cumsum() # get cummalative sum of the delta column
                df['normalised_delta_t_cum'] = df['delta_t_cum'] / total_num_seconds 
                df['normalised_delta_t'] = df['delta_t'].dt.total_seconds() / total_num_seconds # normalising date time
            
            # course and speed deltas
            df['delta_c'] = (df['course']-df['course'].shift()).abs() # course difference
            df.loc[df['delta_c'] >= 180, 'delta_c'] -= 360 # in degrees so we need to make sure the range stays between 0 and 360
            df['delta_c'] = df['delta_c'].abs()
            df['delta_c'] = df['delta_c'].fillna(0)
            
            # convert lat and lon to x, y, z coordinates as they are 3D
            df['targets'] = df['desired'] # move targets to end
            # df = df.drop(columns=['desired'])
            df = df.drop(columns=drop_columns, inplace=True)
            
        return list_df
    
    # =============================================================================
    # helper method that calls segment_dataframes
    # =============================================================================
    def build_sequences(self, time_period, threshold, drop_columns):
        list_df = self.partition_vessels()
        df_list_seq = []
        
        # first, segment the datframes
        for df in list_df:    
            df_list_seq.append(self.segment_dataframe(dataframe=df, time_period=time_period, threshold=threshold))
            
        # now calculate deltas
        for list_df in df_list_seq:
            list_df = self.create_deltas(list_df=list_df, interpolated=False, drop_columns=drop_columns)
        
        return df_list_seq

        
        
    # =============================================================================
    # method to flatten a list of lists of dataframes
    # =============================================================================
    def flatten_df_list(self, list_df, nested_list):
        total_list = []
        mask = []

        # if the list_df argument is 3 dimensional
        if nested_list:
            for sublist in list_df:
                total_list.append(pd.concat(sublist))
            df = pd.concat(total_list) # flattened dataframe of all the segments, normalise this and then convert back to segments
            
            # need to create a mask that determines the size of the sequence length
            for sublist in list_df:
                for df_temp in sublist:
                    mask.append(len(df_temp))
                    
        # if the list_df argument is one 2 dimensional
        else:
            df = pd.concat(list_df)
            for df_temp in list_df:
                mask.append(len(df_temp))
            
        return df, mask


    # =============================================================================
    # convert single dataframe to segments using saved mask
    # =============================================================================
    def re_segment(self, df, mask, dataframe):
        # convert back to segments
        final_df_list = []
        df_idx = 0
        for i in range(len(mask)):
            if dataframe:
                final_df_list.append(df.iloc[df_idx:df_idx + mask[i], :])
            else:
                final_df_list.append(df[df_idx:df_idx + mask[i], :])
            df_idx += mask[i]
        return final_df_list


    # =============================================================================
    # helper method that splits the data into train and test sets and then normalises based on the data in the train sets
    # =============================================================================
    def split(self, data, train_ratio): # takes as input the output of create_deltas()
        
        # shuffle must be false as we need to preserve the order of the time sequences 
        data_train, data_rem = train_test_split(data, train_size=train_ratio, random_state=None, shuffle=False, stratify=None)
        data_valid, data_test = train_test_split(data_rem, test_size=0.5, random_state=None, shuffle=False, stratify=None)

        return data_train, data_valid, data_test

    
    # =============================================================================
    # @output returns a single dataframe of interpolated vessels
    # @params - vessel list is output of partition_vessels() method, target interval is the size of the time steps
    # =============================================================================
    def linear_interpolation(self, vessel_list, target_interval, visualise=False):
        drop_columns = ['timestamp', 'mmsi', 'distance_from_shore', 'distance_from_port', 'is_fishing']
    
        # prepare master dataframe
        df_ex = vessel_list[0]
        df = pd.DataFrame().reindex_like(df_ex).dropna()
        df.drop(columns=drop_columns, inplace=True)
    
        df_orig = df.copy()
        del df_ex # keep memory clean
    
        for v in vessel_list:
            # =============================================================================
            # Resample each vessel
            # =============================================================================
            v.index = v['timestamp'] # set index to timestamp
            v.index.name = 'index'
            v.drop(columns=drop_columns, inplace=True)
            v = v[~v.index.duplicated(keep='first')]
            resampled_v = v.resample(str(target_interval) + 'T').mean()
            
            # =============================================================================
            # Linear interpolation
            # =============================================================================
            resampled_v = resampled_v.interpolate(method='linear')
            
            # =============================================================================
            # Combine vessels into master dataframe
            # =============================================================================
            df = pd.concat([df, resampled_v])
            df_orig = pd.concat([df_orig, v])
            
            # =============================================================================
            # Visualise the difference between the intepolated and original trajectories
            # =============================================================================
            if visualise:
                plt.plot(v['lat'], v['lon'])
                plt.show()
                plt.plot(resampled_v['lat'], resampled_v['lon'], c='green')
                plt.show()
    
        return df




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
    def return_size(self):
        return len(self.build_sequences())



