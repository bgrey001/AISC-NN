#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 19:34:51 2022

@author: BenedictGrey

Data pre-processing script for purse seiners class




"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import the one hot encoded classes and find the appropriate row
df_classes = pd.read_csv('data/one_hot_encoded_classes.csv')
class_row = df_classes.loc[df_classes['purse_seines'] == 1]


# import trollers dataset, duplicate rows for labels and append them to dataframe, finally drop useless source feature
df = pd.read_csv('data/purse_seines.csv')
df_seines_label = pd.concat([class_row] * len(df), ignore_index=True)
df = df.join(df_seines_label, how='left')
df = df.drop(columns=['source'])



# ---------------------------- time series investigation for visualisation ----------------------------


from pattern_visualisation import segment_dataframe
    
print(len(df))

df_temp = segment_dataframe(df, 24, len(df))
len(df_temp)

print(df_temp[0])

num = 0
plt.scatter(df_temp[num]['lat'], df_temp[num]['lon'], s=15)
plt.plot(df_temp[num]['lat'], df_temp[num]['lon'])

# ---------------------------- visualisation for vessels in whole dataset ----------------------------


from pattern_visualisation import plot_vessels

df_temp2 = plot_vessels(df, len(df))

num = 5
plt.scatter(df_temp2[num]['lat'], df_temp2[num]['lon'], s=1)
plt.plot(df_temp2[num]['lat'], df_temp2[num]['lon'])





# get all vessels in the set that are fishing
df_is_fishing = df.loc[df['is_fishing'] > 0]
print(len(df_is_fishing))


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



# find unique values in the mmsi column aka find how many individual vessels are in this set
print(df_is_fishing.mmsi.value_counts())
a = df['mmsi'].unique()
print(len(a))


# select independent and dependant variables
X = df.iloc[:, :-6].values
y = df.iloc[:, (len(df.columns) - 6):]
