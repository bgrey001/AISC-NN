#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:46:15 2022

@author: BenedictGrey

Data pre-processing script for trawlers class


DATA_SIZE = WIDTH = FEATURES
            HEIGHT = TIME-STEPS
            
            rows=features, columns=timesteps

"""

# chapter 10, 11 chollet, transformers 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the one hot encoded classes and find the appropriate row
df_classes = pd.read_csv('data/one_hot_encoded_classes.csv')
class_row = df_classes.loc[df_classes['trawlers'] == 1]


# import trollers dataset, duplicate rows for labels and append them to dataframe, finally drop useless source feature
df = pd.read_csv('data/trawlers.csv')
df_trawlers_label = pd.concat([class_row] * len(df), ignore_index=True)
df = df.join(df_trawlers_label, how='left')
df = df.drop(columns=['source'])


# get all vessels in the set that are fishing
df_is_fishing = df.loc[df['is_fishing'] > 0]
print(df_is_fishing)

# find unique mmsi's
vector_mmsi = df_is_fishing['mmsi'].unique()
print(len(vector_mmsi)) # how many unique vessels in this set
print(vector_mmsi[10:20])


# what if instead of only using different vessels for training we use time windows
# e.g, only 37 unique vessels in this dataset, but if we use time windows of say 24hours and then divide the dateset based on this time window and then train the dataset based on that time window

m = plot_vessels(df_is_fishing, vector_mmsi, 50)
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

plt.scatter(m[5]['lat'], m[5]['lon'])
plt.plot(m[5]['lat'], m[5]['lon'])

plt.scatter(m[6]['lat'], m[6]['lon'])
plt.plot(m[6]['lat'], m[6]['lon'])

plt.scatter(m[7]['lat'], m[7]['lon'])
plt.plot(m[7]['lat'], m[7]['lon'])

plt.scatter(m[8]['lat'], m[8]['lon'])
plt.plot(m[8]['lat'], m[8]['lon'])

plt.scatter(m[9]['lat'], m[9]['lon'])
plt.plot(m[9]['lat'], m[9]['lon'])

# find and decode timestamps to partition the data into 24hr time periods for testing, ignoring mmsi

print(df['timestamp'])

# find unique values in the mmsi column aka find how many individual vessels are in this set
print(df_is_fishing.mmsi.value_counts())
a = df['mmsi'].unique()
print(len(a))

# select independent and dependant variables
X = df.iloc[:, :-6].values
y = df.iloc[:, (len(df.columns) - 6):]
