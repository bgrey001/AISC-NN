#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 19:58:59 2022

@author: BenedictGrey

Interpolation script for data pre-processing


CURRENT METHODOLOGY:
    1) Convert each interpolated time segment (currently a dataframe with a certain number of entries) into a single vector of length of number of time steps
    2) Feed the vector into a 1D CNN for classification
    
ALTERNATIVE:
    1) Process the time sequences into certain lengths (24hrs for example)
    2) Capture these as images for CNN and then feed into 2D CNN
    
    
OBSTACLES:
    1) Figure out how to process data into the vector for input
        What happens to the other features? 
        How can we squash all time steps into one vector?
        All we need is the time steps (in a vector), lat and lon and desired value
        Trim the dataframe of anything redundant? Maybe keep these
    2) Find another method (speed perhaps) as a means for determining whether the vessels are fishing
        Due to the fact that the is_fishing has too many outliers
         

Implementation steps:

    1) Break down the dataset into each mmsi vessel for continuity
    2) Use upsampling to create feature
    3) Fill in the empty new feature with interpolated features
    3) Create regular time series intervals for each vessel using linear regression from pandas
    
    
    

NOTE: Measuring skewness of distribution of the timestamps in the data
Skewness measures the asymmetry in the data

Script details:
        
    1. Divide dataset by mmsi using create_set()
    2. Interpolated each vessel dataset using interpolate_vessels()
    3. Divide aggregate interpolated datasets into one tensor containing batches of 24hrs using initialise_tensor()
    

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime, timedelta


# test dataset using trollers as its small
df = pd.read_csv('data/trollers.csv')
# convert to readable timestamp
df['timestamp'] = pd.to_datetime(df.timestamp, unit='s')
df = df.drop(columns=['source'])
df['pseudo'] = 0


# step 1, divide the dataset based on mmsi
def create_set(dataframe):
    vessel_list = []
    vessel_ids = dataframe['mmsi'].unique()
    for v in vessel_ids:
        temp_df = dataframe[dataframe['mmsi'] == v]
        vessel_list.append(temp_df)
    return vessel_list

# call the function to create set of dfs of vessels
vessel_list = create_set(df)


# try with timesteps of 1 hour and see how it inflates the dataset
def interpolate_vessels(vessel_list, target_interval):
    
    target_interval = timedelta(minutes=target_interval)
    list_output = []

    total_time_elapsed = timedelta(hours=0)
    total_average = timedelta(hours=0)
    
    for v in vessel_list:
        
        t0 = v['timestamp'].iloc[0]
        tn = v['timestamp'].iloc[len(v) - 1]
        time_elapsed = tn - t0
        average_interval = time_elapsed / len(v)
        
        total_time_elapsed = time_elapsed + total_time_elapsed
        total_average = total_average + average_interval

        new_range = pd.date_range(start=t0, end=tn, freq=target_interval)
        df_time = new_range.to_frame().reset_index()

        df_time = df_time.rename(columns={0: 'timestamp'})
        df_time = df_time.drop(columns='index')
        
        # column to represent whether the row is real or interpolated
        #df_time['pseudo'] = 1
        
        # join original and regular timestep frames
        frames = [v, df_time]
        df_total = pd.concat(frames)
        df_total = df_total.sort_values(by=['timestamp'])

        df_total['pseudo'] = df_total['pseudo'].fillna(1)

        # solution for linear interpolation
        output_df = pd.DataFrame()
        output_df['lat'] = df_total['lat'].interpolate(method='linear', limit_direction="both")
        output_df['lon'] = df_total['lon'].interpolate(method='linear', limit_direction="both")
        output_df['distance_from_shore'] = df_total['distance_from_shore'].interpolate(method='linear', limit_direction="both")
        output_df['distance_from_port'] = df_total['distance_from_port'].interpolate(method='linear', limit_direction="both")
        output_df['speed'] = df_total['speed'].interpolate(method='linear', limit_direction="both")


        # add additional data to batch
        output_df['pseudo'] = df_total['pseudo']
        
        output_df['desired'] = 
        
    
        # y should be (size, batch, 1) for integer encoding
        
        # drop anything that has pseudo 0
        output_df.drop(output_df[output_df['pseudo'] != 1].index, inplace = True)
        output_df = output_df.drop(columns=['pseudo'])
        
        dataframe = df_total
        
        print("length of interpolated:",len(output_df), "length of original:", len(v))

        # drop anything with pseudo value of 0
        #df_total = df_total.drop(df_total[df_total['pseudo'].index] != 1)
        
        #return output_df
        # now we need to remove the original irregular time steps, denoted by a pseudo value as 0
        list_output.append(np.array(output_df.values)) # proper return value
        #list_output.append(df_total)
    
    #total_average = total_average/5 # use to see average of time steps in the dataset
    return list_output, dataframe


# break into 24 hour segments by taking every 48 time steps and putting them in a batch
def initialise_tensor(interpolated_vessels, t_steps):
    batches = []
    for v in interpolated_vessels:
        for i in range(0, len(v), t_steps):
            # if the end of the vessel sequence isn't long enough we need to take some other values
            if len(v[i:(i+t_steps), :]) < t_steps:
                length_batch = len(v[i:(i+t_steps), :])
                diff = t_steps - length_batch
                batches.append(v[i-diff:(i+t_steps), :])
            else:    
                batches.append(v[i:(i+t_steps), :])
                
    return torch.tensor(batches)
    #return batches


time_step = 15
batch_size = 1440 / time_step

iv = interpolate_vessels(vessel_list, time_step) # using 30 minute intervals this means we get 48 timesteps per 24 hours

batches = initialise_tensor(interpolated_vessels=iv, t_steps=batch_size)






vessel = batches[1100]
len(v)

plt.plot(v[:, 0], v[:, 1])
plt.scatter(v[:, 0], v[:, 1], s=1)



# ------------------------ visual functions ------------------------

# visualise vessel plots
def vis_vessels(range_start, range_end, all_vessels):
    if (all_vessels == False):
        v_sub = vessel_list[range_start: range_end]
    else:
        v_sub = vessel_list
    for v in v_sub:
        plt.plot(v['lat'], v['lon'])
        plt.scatter(v['lat'], v['lon'], s=15)
        
vis_vessels(range_start=3, range_end=4, all_vessels=False)    

    
# show differences in real dataset and interpolated lengths
def print_size_differences(vessel_list, interpolated_vessels):
    for i in range(0, len(vessel_list)): 
        print(len(interpolated_vessels[i]), len(vessel_list[i]))
        
print_size_differences(vessel_list, iv)


# ------------------------ old -------------------------------

for b in batches:
    if (len(b) != 48):
        print(len(b))
    

example = batches[8000]
plt.scatter(example[:, 0], example[:, 1], s=5)
plt.plot(example[:, 0], example[:, 1])

# total timesteps = 166243
# lets interpolate time positions every 60 seconds


# option 1: find how much time has elapsed and then divide that by number of entries and that is out time difference steps (linear)


# this segment creates the new dataframe containing the evenly spaced time steps

length = len(df)
print(length)

#t0 = datetime.fromtimestamp(df['timestamp'].iloc[0])
#tn = datetime.fromtimestamp(df['timestamp'].iloc[len(df) - 1])
t0 = df['timestamp'].iloc[0]
tn = df['timestamp'].iloc[len(df) - 1]

total_time_elapsed = df['timestamp'].iloc[len(df) - 1] - df['timestamp'].iloc[0]
total_time_elapsed = tn - t0

average_interval = total_time_elapsed / len(df)

print(average_interval)
print(total_time_elapsed)

# create new column with time range calculated

new_range = pd.date_range(start=t0, end=tn, freq=average_interval)
df_time = new_range.to_frame().reset_index()

df_time = df_time.rename(columns={0: 'timestamp'})
df_time = df_time.drop(columns='index')



# add new timestamps into the main dataframe and then impute values, finally remove irregular time steps


df_time['regular'] = 1
df['regular'] = 0


# concat dataframes

frames = [df, df_time]
df_total = pd.concat(frames)
df_total = df_total.sort_values(by=['timestamp'])




# Taking care of missing data

# using pandas interpolate
df_temp = df_total['lat'].interpolate(method='linear', limit_direction="both")
df_temp2 = df_total['lon'].interpolate(method='linear', limit_direction="both")




# solution for linear interpolation
df2 = pd.DataFrame()
df2['lat'] = df_total['lat'].interpolate(method='linear', limit_direction="both")
df2['lon'] = df_total['lon'].interpolate(method='linear', limit_direction="both")

plt.scatter(df2['lat'], df2['lon'], s=5)
plt.plot(df2['lat'], df2['lon'])

# trying 'time' method for interpolation


df2 = pd.DataFrame()
df2['lat'] = df_total['lat'].interpolate(method='linear', limit_direction="both")
df2['lon'] = df_total['lon'].interpolate(method='linear', limit_direction="both")

plt.scatter(df2['lat'], df2['lon'], s=5)
plt.plot(df2['lat'], df2['lon'])

#upsampled = series.resample('D')
#interpolated = upsampled.interpolate(method='linear')
#print(interpolated.head(32))

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = df_total.iloc[:, 2:9]
lat_lon = df_total.iloc[:, 6:8].values
x1 = df_total.iloc[:, 2].values
lat = df_total.iloc[:, 6].values

# impute single column
df_imputed_single = imputer.fit_transform(x1.reshape(-1,1))
df_imputed = pd.DataFrame(imputer.fit_transform(X))

imputed_latlon = imputer.fit_transform(lat_lon)

print(imputed_latlon[:, 0])

plt.scatter(imputed_latlon[:,0], imputed_latlon[:,1], s=5)
plt.plot(imputed_latlon[:,0], imputed_latlon[:,1])

# impute multivariate
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)

df_multi_imputed = imp.fit_transform(lat_lon)
plt.scatter(df_multi_imputed[:, 0], df_multi_imputed[:,1], s=5)
plt.plot(df_multi_imputed[:,0], df_multi_imputed[:,1])

