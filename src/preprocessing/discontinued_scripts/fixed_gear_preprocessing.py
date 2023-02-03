#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:42:58 2022

@author: BenedictGrey

Data pre-processing script for fixed gear class

t is height (number of time steps) and k is number of features

RNN - LSTM

Attention

"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import the one hot encoded classes and find the appropriate row
df_classes = pd.read_csv('data/one_hot_encoded_classes.csv')
class_row = df_classes.loc[df_classes['fixed_gear'] == 1]


# import trollers dataset, duplicate rows for labels and append them to dataframe, finally drop useless source feature
df = pd.read_csv('data/fixed_gear.csv')
df_seines_label = pd.concat([class_row] * len(df), ignore_index=True)
df = df.join(df_seines_label, how='left')
df = df.drop(columns=['source'])

# ---------------------------- visualisation for segments of time for vessels ----------------------------


from pattern_visualisation import segment_dataframe
    
print(len(df))

df_temp = segment_dataframe(dataframe=df, time_period=24, sample_size=len(df), threshold=35)
len(df_temp)

print(df_temp[0])

num = 150
plt.scatter(df_temp[num]['lat'], df_temp[num]['lon'], s=15)
plt.plot(df_temp[num]['lat'], df_temp[num]['lon'])




# ---------------------------- visualisation for vessels in whole dataset ----------------------------


from pattern_visualisation import plot_vessels

df_temp2 = plot_vessels(df, len(df))
len(df_temp2)

num = 15
plt.scatter(df_temp2[num]['lat'], df_temp2[num]['lon'], s=1)
plt.plot(df_temp2[num]['lat'], df_temp2[num]['lon'])