#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:47:31 2022

@author: BenedictGrey

Data pre-processing script for trollers class

IMPORTANT NOTE: is_fishing label is not accurate as there are a huge number of datapoints with no value

Note to self: use integer encoding instead of one hot encoding as there are 6 different classes and it's computationally heavy to use one hot encoding

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the one hot encoded classes and find the appropriate row
df_classes = pd.read_csv('data/one_hot_encoded_classes.csv')
class_row = df_classes.loc[df_classes['trollers'] == 1]

# import trollers dataset, duplicate rows for labels and append them to dataframe, finally drop useless source feature
df = pd.read_csv('data/trollers.csv')
df_trollers_label = pd.concat([class_row] * len(df), ignore_index=True)
df = df.join(df_trollers_label, how='left')
df = df.drop(columns=['source'])





# ---------------------------- time series investigation for visualisation ---------------------------- #


from pattern_visualisation import segment_dataframe
    
print(len(df))

df_temp = segment_dataframe(dataframe=df, time_period=24, threshold=25, sample_size=len(df))
len(df_temp[0])



num = 10
plt.scatter(df_temp[num]['lat'], df_temp[num]['lon'], s=15)
plt.plot(df_temp[num]['lat'], df_temp[num]['lon'])










# select independent and dependant variables
X = df.iloc[:, :-6].values
y = df.iloc[:, (len(df.columns) - 6):]

# get all vessels in the set that are fishing
df_is_fishing = df.loc[df['is_fishing'] > 0]
print(len(df_is_fishing))

# find unique mmsi's
vector_mmsi = df_is_fishing['mmsi'].unique()

from pattern_visualisation import plot_vessels

m = plot_vessels(df_is_fishing, 50)
print(m[0])

plt.scatter(m[0]['lat'], m[0]['lon'])
plt.plot(m[0]['lat'], m[0]['lon'])

plt.scatter(m[1]['lat'], m[1]['lon'])
plt.plot(m[1]['lat'], m[1]['lon'])

plt.scatter(m[2]['lat'], m[2]['lon'])
plt.plot(m[2]['lat'], m[2]['lon'])

plt.scatter(m[3]['lat'], m[3]['lon'])
plt.plot(m[3]['lat'], m[3]['lon'])

plt.scatter(m[4]['lat'], m[4]['lon'])
plt.plot(m[4]['lat'], m[4]['lon'])

# ---------------------------- time series investigation for visualisation ---------------------------- #

from pattern_visualisation import segment_dataframe
    
print(len(df_is_fishing))
df_temp = segment_dataframe(df_is_fishing, 24, len(df_is_fishing))
print(len(df_temp))
num = 21
plt.scatter(df_temp[num]['lat'], df_temp[num]['lon'], s=20)
plt.plot(df_temp[num]['lat'], df_temp[num]['lon'])

# plot all datapoints for one ship
unique_vessels = df_is_fishing['mmsi'].unique()
df_ind = df[df['mmsi'] == unique_vessels[2]]
plt.scatter(df_ind['lat'], df_ind['lon'], s=0.5)
plt.plot(df_ind['lat'], df_ind['lon'])

# find unique values in the mmsi column aka find how many individual vessels are in this set
print(df_is_fishing.mmsi.value_counts())
a = df['mmsi'].unique()
print(len(a))





