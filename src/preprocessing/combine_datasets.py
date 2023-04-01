#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:48:15 2022
@author: benedict

Combining datasets for varying time sequences/preprocessing script utilising ppm

Data fields for:
    Varying time series:
        speed | lat | lon | delta_time | delta_course | target
    Linearly interpolated time series:
        speed | course | lat | lon | desired
    Varying for non-linear attention interpolation:
        speed | lat | lon | delta_time_cum | delta_course | target

"""
# =============================================================================
# dependencies
# =============================================================================
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import dask.dataframe as dd
import pickle
import random
import math
import os
from tqdm import tqdm
from datetime import datetime
from pycm import ConfusionMatrix
from dask.dataframe import from_pandas

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

import pre_processing_module as ppm
    
random.seed(42)

# =============================================================================
# options: 'varying', 'linear_interp', 'non_linear', 'feature_extraction'
# =============================================================================
choice = 'linear_interp'
saving = True
n_classes = 6

troller_module = ppm.pre_processing_module('trollers')
pole_module = ppm.pre_processing_module('pole_and_line') 
purse_module = ppm.pre_processing_module('purse_seines') 
fixed_module = ppm.pre_processing_module('fixed_gear') 
trawlers_module = ppm.pre_processing_module('trawlers') 
drifting_module = ppm.pre_processing_module('drifting_longlines') 

match choice: 

    case 'varying':
    # =============================================================================
    # varying sequence lengths, padding will take place in the data loader module
    # =============================================================================
        save_version = 4 # latest version is 4
        n_features = 5
        time_period = 24
        threshold = 1
        
        aggregate_mask = []
        drop_columns = ['mmsi', 'distance_from_shore', 'distance_from_port', 'delta_t_cum', 'course', 'is_fishing', 'delta_t', 'desired', 'timestamp']
        
        # trollers, order matters here!
        troller_list_df = troller_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        troller_df, troller_mask = troller_module.flatten_df_list(troller_list_df, nested_list=True)
        aggregate_mask += troller_mask
        aggregate_df = troller_df
        
        # # pole and line
        pole_list_df = pole_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        pole_df, pole_mask = pole_module.flatten_df_list(pole_list_df, nested_list=True)
        aggregate_mask += pole_mask
        aggregate_df = pd.concat([aggregate_df, pole_df], ignore_index=True, axis=0)
        
        # purse seines
        purse_list_df = purse_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        purse_df, purse_mask = purse_module.flatten_df_list(purse_list_df, nested_list=True)
        aggregate_mask += purse_mask
        aggregate_df = pd.concat([aggregate_df, purse_df], ignore_index=True, axis=0)
        
        # fixed gear
        fixed_list_df = fixed_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        fixed_df, fixed_mask = fixed_module.flatten_df_list(fixed_list_df, nested_list=True)
        aggregate_mask += fixed_mask
        aggregate_df = pd.concat([aggregate_df, fixed_df], ignore_index=True, axis=0)
        
        # trawlers
        trawlers_list_df = trawlers_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        trawlers_df, trawlers_mask = trawlers_module.flatten_df_list(trawlers_list_df, nested_list=True)
        aggregate_mask += trawlers_mask
        aggregate_df = pd.concat([aggregate_df, trawlers_df], ignore_index=True, axis=0)
        
        # drifting longlines
        drifting_list_df = drifting_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        drifting_df, drifting_mask = drifting_module.flatten_df_list(drifting_list_df, nested_list=True)
        aggregate_mask += drifting_mask
        aggregate_df = pd.concat([aggregate_df, drifting_df], ignore_index=True, axis=0)
    
        
        # instantiate empty preprocessing module for the operations to follow
        module = ppm.pre_processing_module("")
        
        # rebuild the sequences using the mask, then we shuffle and create more masks
        df_list = module.re_segment(aggregate_df, aggregate_mask, dataframe=True)

        # find max sequence length for padding process
        total_ = 0
        seq_length = 0
        for item in df_list:
            total_ += len(item)
            if len(item) > seq_length:
                seq_length = len(item) # RESULT: 2931


        # now shuffle
        random.shuffle(df_list)

        # split the data while it's in batches
        batch_train, batch_valid, batch_test = module.split(df_list, train_ratio=(0.8))
        total = len(aggregate_df)
        
        # now we flatten the batches
        batch_train_flat, batch_train_mask = module.flatten_df_list(batch_train, nested_list=False)
        batch_valid_flat, batch_valid_mask = module.flatten_df_list(batch_valid, nested_list=False)
        batch_test_flat, batch_test_mask = module.flatten_df_list(batch_test, nested_list=False)
        # print(len(batch_train_flat) + len(batch_valid_flat) + len(batch_test_flat)) # only 3 off the original size of the data 
        
        # now we normalise
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        
        # covert dataframes to arrays
        data_train = batch_train_flat.values 
        data_valid = batch_valid_flat.values
        data_test = batch_test_flat.values
        
        # scale
        scale_train = scaler.fit_transform(data_train[:, [0, 1, 2, 4]])
        scale_valid = scaler.transform(data_valid[:, [0, 1, 2, 4]])
        scale_test = scaler.transform(data_test[:, [0, 1, 2, 4]])
        # print(len(scale_train) + len(scale_valid) + len(scale_test)) # only 3 off the original size of the data 
        
        # convert nans to prevent vanishing gradient
        scale_train = np.nan_to_num(scale_train)
        scale_valid = np.nan_to_num(scale_valid)
        scale_test = np.nan_to_num(scale_test)
        
        # add normalised columns back to data arrays, index of 3 is the already normalised time delta which needed custom normalisation
        data_train[:, [0, 1, 2, 4]] = scale_train
        data_valid[:, [0, 1, 2, 4]] = scale_valid
        data_test[:, [0, 1, 2, 4]] = scale_test
        # print(len(data_train) + len(data_valid) + len(data_test)) # only 3 off the original size of the data 
        
        # rebuild time sequences
        data_train_seqs = module.re_segment(data_train, batch_train_mask, dataframe=False)
        data_valid_seqs = module.re_segment(data_valid, batch_valid_mask, dataframe=False)
        data_test_seqs = module.re_segment(data_test, batch_test_mask, dataframe=False)
        

            
            

    case 'linear_interp':
    # =============================================================================
    # linear interpolation   
    # =============================================================================
        save_version = 6 # latest version is 4
        n_features = 4
        module = ppm.pre_processing_module("")
        
        troller_vessel_list = troller_module.partition_vessels()
        pole_vessel_list = pole_module.partition_vessels()
        purse_vessel_list = purse_module.partition_vessels()
        fixed_vessel_list = fixed_module.partition_vessels()
        trawlers_vessel_list = trawlers_module.partition_vessels()
        drifting_vessel_list = drifting_module.partition_vessels()
    
        target_interval = 60
        
        # join datasets together
        df = pd.concat([
                        module.linear_interpolation(troller_vessel_list, target_interval=target_interval),
                        module.linear_interpolation(pole_vessel_list, target_interval=target_interval),
                        module.linear_interpolation(purse_vessel_list, target_interval=target_interval),
                        module.linear_interpolation(fixed_vessel_list, target_interval=target_interval),
                        module.linear_interpolation(trawlers_vessel_list, target_interval=target_interval),
                        module.linear_interpolation(drifting_vessel_list, target_interval=target_interval),
                        ]).values
        
        print('Successfully concatenated all dataframes')
    
        # =============================================================================
        # Segment into 24 hours sequences
        # 1440 minutes in a day, therefore the total length of the seqence should be 1440 / target_interval
        # =============================================================================
        seq_length = 1440 / target_interval
        def linear_segment(data):
            return np.array_split(data, math.ceil(data.shape[0]/seq_length))
    
        # convert to sequences
        list_seq = linear_segment(df)
        del df # clean
    
        # shuffle
        random.shuffle(list_seq)
    
        # =============================================================================
        # Split into train and test sets
        # =============================================================================
        data_train, data_valid, data_test = module.split(data=list_seq, train_ratio=0.8)
        del list_seq # clean
    
        # =============================================================================
        # Flatten back into respective matrices
        # =============================================================================
        train_flat = np.vstack(data_train)
        valid_flat = np.vstack(data_valid)
        test_flat = np.vstack(data_test)
        del data_train, data_valid, data_test # clean
    
        # =============================================================================
        # 6) Scale the data based on the values calculated on train data
        # =============================================================================
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
    
        # scale
        scale_train = scaler.fit_transform(train_flat[:, [0, 1, 2, 3]])
        scale_valid = scaler.transform(valid_flat[:, [0, 1, 2, 3]])
        scale_test = scaler.transform(test_flat[:, [0, 1, 2, 3]])
    
        # add scaled columns back to data arrays
        train_flat[:, [0, 1, 2, 3]] = scale_train
        valid_flat[:, [0, 1, 2, 3]] = scale_valid
        test_flat[:, [0, 1, 2, 3]] = scale_test
        del scale_train, scale_valid, scale_test
    
        data_train_seqs = linear_segment(train_flat)
        data_valid_seqs = linear_segment(valid_flat)
        data_test_seqs = linear_segment(test_flat)
        del train_flat, valid_flat, test_flat





    # =============================================================================
    # non-linear interpolation preprocessing
    # =============================================================================
    case 'non_linear':
        save_version = 1 # latest version is none
        n_features = 5
        time_period = 24
        threshold = 1
        
        aggregate_mask = []
        drop_columns = ['mmsi', 'distance_from_shore', 'distance_from_port', 'delta_t_cum', 'normalised_delta_t', 'course', 'is_fishing', 'delta_t', 'desired', 'timestamp']
        
        # trollers, order matters here!
        troller_list_df = troller_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        troller_df, troller_mask = troller_module.flatten_df_list(troller_list_df, nested_list=True)
        aggregate_mask += troller_mask
        aggregate_df = troller_df
        
        # # pole and line
        pole_list_df = pole_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        pole_df, pole_mask = pole_module.flatten_df_list(pole_list_df, nested_list=True)
        aggregate_mask += pole_mask
        aggregate_df = pd.concat([aggregate_df, pole_df], ignore_index=True, axis=0)
        
        # purse seines
        purse_list_df = purse_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        purse_df, purse_mask = purse_module.flatten_df_list(purse_list_df, nested_list=True)
        aggregate_mask += purse_mask
        aggregate_df = pd.concat([aggregate_df, purse_df], ignore_index=True, axis=0)
        
        # fixed gear
        fixed_list_df = fixed_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        fixed_df, fixed_mask = fixed_module.flatten_df_list(fixed_list_df, nested_list=True)
        aggregate_mask += fixed_mask
        aggregate_df = pd.concat([aggregate_df, fixed_df], ignore_index=True, axis=0)
        
        # trawlers
        trawlers_list_df = trawlers_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        trawlers_df, trawlers_mask = trawlers_module.flatten_df_list(trawlers_list_df, nested_list=True)
        aggregate_mask += trawlers_mask
        aggregate_df = pd.concat([aggregate_df, trawlers_df], ignore_index=True, axis=0)
        
        # drifting longlines
        drifting_list_df = drifting_module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
        drifting_df, drifting_mask = drifting_module.flatten_df_list(drifting_list_df, nested_list=True)
        aggregate_mask += drifting_mask
        aggregate_df = pd.concat([aggregate_df, drifting_df], ignore_index=True, axis=0)
        
        
        # instantiate empty preprocessing module for the operations to follow
        module = ppm.pre_processing_module("")
        
        # rebuild the sequences using the mask, then we shuffle and create more masks
        df_list = module.re_segment(aggregate_df, aggregate_mask, dataframe=True)


        # find max sequence length for padding process
        total_ = 0
        seq_length = 0
        for item in df_list:
            total_ += len(item)
            if len(item) > seq_length:
                seq_length = len(item) # RESULT: 2931


        # now shuffle
        random.shuffle(df_list)

        # split the data while it's in batches
        batch_train, batch_valid, batch_test = module.split(df_list, train_ratio=(0.8))
        total = len(aggregate_df)
        
        # now we flatten the batches
        batch_train_flat, batch_train_mask = module.flatten_df_list(batch_train, nested_list=False)
        batch_valid_flat, batch_valid_mask = module.flatten_df_list(batch_valid, nested_list=False)
        batch_test_flat, batch_test_mask = module.flatten_df_list(batch_test, nested_list=False)
        # print(len(batch_train_flat) + len(batch_valid_flat) + len(batch_test_flat)) # only 3 off the original size of the data 
        
        # now we normalise
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        
        # covert dataframes to arrays
        data_train = batch_train_flat.values 
        data_valid = batch_valid_flat.values
        data_test = batch_test_flat.values
        
        # scale
        scale_train = scaler.fit_transform(data_train[:, [0, 1, 2, 4]])
        scale_valid = scaler.transform(data_valid[:, [0, 1, 2, 4]])
        scale_test = scaler.transform(data_test[:, [0, 1, 2, 4]])
        # print(len(scale_train) + len(scale_valid) + len(scale_test)) # only 3 off the original size of the data 
        
        # convert nans to prevent vanishing gradient
        scale_train = np.nan_to_num(scale_train)
        scale_valid = np.nan_to_num(scale_valid)
        scale_test = np.nan_to_num(scale_test)
        
        # add normalised columns back to data arrays, index of 3 is the already normalised time delta which needed custom normalisation
        data_train[:, [0, 1, 2, 4]] = scale_train
        data_valid[:, [0, 1, 2, 4]] = scale_valid
        data_test[:, [0, 1, 2, 4]] = scale_test
        # print(len(data_train) + len(data_valid) + len(data_test)) # only 3 off the original size of the data 
        
        # rebuild time sequences
        data_train_seqs = module.re_segment(data_train, batch_train_mask, dataframe=False)
        data_valid_seqs = module.re_segment(data_valid, batch_valid_mask, dataframe=False)
        data_test_seqs = module.re_segment(data_test, batch_test_mask, dataframe=False)



    # =============================================================================
    # feature extraction preprocessing
    # =============================================================================
    case 'feature_extraction':
        save_version = 2 # latest version is none
        # =============================================================================
        # load data
        # starting with the standardised unpadded varying time sequences 
        # =============================================================================
        choice ='varying'
        version = '3'
        def load_sequences(split):
            with open(f'../../data/pkl/{choice}/{split}_v{version}.pkl', 'rb') as f:
                seq_list = pickle.load(f)
            with open(f'../../data/pkl/{choice}/utils_v{version}.pkl', 'rb') as f:
                obj = pickle.load(f)
                n_features = obj[0]
                n_classes = obj[1]
                seq_length = obj[2]
                
            return seq_list, n_features, n_classes, seq_length

        train_seq_list, _, _, _ = load_sequences('train')
        test_seq_list, _, _, _ = load_sequences('test')
        valid_seq_list, _, _, _ = load_sequences('valid')
        seq_list = train_seq_list + test_seq_list + valid_seq_list

        # testing splits of data, didn't use stratified split due to size of data and the results suggest this was the right call, very even split
        def test_sample_size(train_seq_list, test_seq_list, valid_seq_list):
            train_arr = np.concatenate(train_seq_list)
            df1 = pd.DataFrame(train_arr)
            res1 = df1[5].value_counts(normalize=True)
            
            test_arr = np.concatenate(test_seq_list)
            df2 = pd.DataFrame(test_arr)
            res2 = df2[5].value_counts(normalize=True)
            
            valid_arr = np.concatenate(valid_seq_list)
            df3 = pd.DataFrame(valid_arr)
            res3 = df3[5].value_counts(normalize=True)

        # =============================================================================
        # extract targets and features from sequences and then perform feature extraction using tsfresh
        # =============================================================================
        def process(seq_list, select, save, save_version):
            
            out_df = pd.DataFrame(columns = ['id','time','f1', 'f2', 'f3', 'f4', 'f5'])
            targets = []
            
            for idx, seq in enumerate(tqdm(seq_list)):
                seq_feat = seq[:, :-1]
                seq_targ = seq[:, -1][0]
                targets.append(seq_targ)
                
                sub_df = pd.DataFrame(seq_feat, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
                sub_df.insert(loc=0, column='time', value=sub_df.index)
                sub_df.insert(loc=0, column='id', value=idx)
                
                out_df = pd.concat([out_df, sub_df])
                # if idx == 1500: break
                
            targets = pd.Series(targets)
            
            # extract features
            extraction_settings = ComprehensiveFCParameters() # this can be tuned?
            
            # each sequence is being pushed into a single row with all features extracted along the y axis
            X_fe = extract_features(out_df, column_id='id', column_sort='time', default_fc_parameters=extraction_settings, impute_function=impute)
            
            # feature selection
            if select:
                X_filt = select_features(X_fe, targets)
                X_filt.head()
            
            # split the data
            X_train, X_test, y_train, y_test = train_test_split(X_fe, targets, test_size=0.4)
            if select: X_train, X_test = X_train[X_filt.columns], X_test[X_filt.columns]
            
            if saving:
                with open(f'../../data/pkl/feature_extraction/v{save_version}.pkl', 'wb') as f:
                    pickle.dump([X_train, X_test, y_train, y_test], f)
                print(f'Saved v{save_version}')
                os._exit()
            
            else: return out_df, targets.reshape(-1, 1)

        if saving:
            process(seq_list=seq_list, select=True, save=True, save_version=save_version)
        
        
        with open('../../data/pkl/feature_extraction/dfs_v1.pkl', 'rb') as f:
            features, targets = pickle.load(f)
            

        ddf = from_pandas(features, npartitions=6)
            
        # extract features
        extraction_settings = MinimalFCParameters() # this can be tuned?

        # each sequence is being pushed into a single row with all features extracted along the y axis
        X_fe = extract_features(ddf, column_id='id', column_sort='time', default_fc_parameters=extraction_settings, disable_progressbar=False)
        X_fe.to_csv('../../data/csv/fe.csv')
        df = X_fe.compute()
        df['Y'] = targets
        df.to_csv('../../data/csv/fe_pd.csv')





# =============================================================================
# save
# =============================================================================
if saving:
    with open(f'../../data/pkl/{choice}/train_v{save_version}.pkl', 'wb') as f:
        pickle.dump(data_train_seqs, f)
    with open(f'../../data/pkl/{choice}/valid_v{save_version}.pkl', 'wb') as f:
        pickle.dump(data_valid_seqs, f)
    with open(f'../../data/pkl/{choice}/test_v{save_version}.pkl', 'wb') as f:
        pickle.dump(data_test_seqs, f)
    with open(f'../../data/pkl/{choice}/utils_v{save_version}.pkl', 'wb') as f:
        pickle.dump([n_features, n_classes, seq_length], f)
    print('Saved files successfully')
        
