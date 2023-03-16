#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:03:45 2023
@author: Benedict Grey

Statistical analysis script
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

import pre_processing_module as ppm
    
random.seed(42)



# =============================================================================
# import data
# =============================================================================
n_features = 5
time_period = 24
threshold = 1

drop_columns = ['normalised_delta_t_cum', 'normalised_delta_t', 'mmsi', 'distance_from_shore', 'distance_from_port', 'delta_t_cum', 'is_fishing', 'desired', 'timestamp']
classes = ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers']
# classes = ['pole_and_line', 'trollers']

agg_list_df = []

for name in classes:
    module = ppm.pre_processing_module(name)
    list_df = module.build_sequences(time_period=time_period, threshold=threshold, drop_columns=drop_columns)
    flat_list = []
    for sublist in list_df: flat_list += sublist
    agg_list_df.append(flat_list)


# =============================================================================
# df = troller_list_df[0][3]
# df['t_hour_pc'] = df['delta_t'].dt.total_seconds() / (60*60)
# df['dist'] = df['t_hour_pc'] * df['speed']
#     
# avg_course_change = df['delta_c'].mean()
# =============================================================================

stats_df = pd.DataFrame(columns=['Class', 
                                'Average speed (knots)', 'Average course change', 'Average distance travelled (miles)', 
                                'std. speed', 'std. course change','std. distance'])

avg_speeds = []
avg_courses = []
dists = []

for idx, class_list in enumerate(agg_list_df):
    for df in tqdm(class_list):    
        # ignore rows where speed is 0 as they will either not be fishing or will be in port
        df = df[df['speed'] != 0]
        if len(df) < 2: continue
        df['t_hour_pc'] = df['delta_t'].dt.total_seconds() / (60*60)
        df['dist'] = df['t_hour_pc'] * df['speed']
        avg_course_change = df['delta_c'].mean()
        # average speed is calculated as the speed at each time step * hour ratio / sum of hour percentages (should be close to 24hrs)
        # print(df)
        avg_speed = sum(df['speed'] * df['t_hour_pc']) / sum(df['t_hour_pc'])
        # print(avg_speed)
        dist = df['dist'].sum()
        
        avg_speeds.append(avg_speed)
        avg_courses.append(avg_course_change)
        dists.append(dist)
        
        
    total_avg_speed = np.mean(avg_speeds)
    total_std_speed = np.std(avg_speeds)
    total_avg_course_change = np.mean(avg_courses)
    total_std_course_change = np.std(avg_courses)
    total_avg_dist = np.mean(dists)
    total_std_dist = np.std(dists)
    row = [f'{classes[idx]}', total_avg_speed, total_std_speed, total_avg_course_change, total_std_course_change, total_avg_dist, total_std_dist]
    stats_df.loc[len(stats_df)] = row

# for seq in troller_list_df:
    
