#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 17:07:07 2022

@author: BenedictGrey

Trajectory comparison file

Idea is to compare the interpolated vessels with raw data (irregular) to try and visualise the difference


"""

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pre_processing_module as ppm


dataset = 'purse_seines'

# =============================================================================
# auto building tensor using ppm class and method build_tensor 
# =============================================================================

module = ppm.pre_processing_module(dataset)
x_tensor, y_tensor = module.build_tensor(time_interval=15, sliding_window=False)

list_df = module.list_df_resampled
df = list_df[0]



# =============================================================================
# visualise batches
# =============================================================================

# =============================================================================
# batches = module.x_batches[:50]
# =============================================================================

# =============================================================================
# module.visualise_batch(batches, 100, False)
# =============================================================================

# =============================================================================
# b = batches[0]
# plt.plot(batch[:,5], batch[:,6])
# plt.show()
# 
# =============================================================================
# =============================================================================
# method to filter out any fishing vessels using course change maximums
# =============================================================================



"""
NORMALISE DATA AND TRY AGAIN!
"""


def isolate_fishing_behaviour(batches, visualise, course_threshold, speed_threshold):
    
    del_indices = []
    n_omissions = 0
    index = 1
    
    for batch in batches:
        
        max_course_diff = 0
        batch_course = batch[:, 4] # get the course column from batch
        batch_average_speed = batch[:, 3].mean() # get the average speed columm from batch
        length_batch = len(batch_course)
        
        for i in range(0, length_batch - 1):
            # find current course and previous course
            current_course = batch_course[i]
            next_course = batch_course[i + 1]
            # find abs differnce between current and previous course
            course_difference = np.absolute(next_course - current_course)
            # course difference is in degrees and therefore anything over 180 degrees has to be converted
            if course_difference >= 180:
                course_difference = 360 - course_difference
            # update max_course_diff to extract the max value    
            if course_difference > max_course_diff:
                max_course_diff = course_difference
            print(course_difference)

                
        # if the max course difference is below this threshold then omit batch from batches
        #if max_course_diff <= course_threshold and batch_average_speed > speed_threshold:
        if max_course_diff <= course_threshold:
            del_indices.append(index)
            n_omissions = n_omissions + 1
        index = index + 1
        
        # visualisation module
        if visualise == True:
            # if the max course difference is below this threshold then omit batch from batches
            #if max_course_diff <= course_threshold and batch_average_speed > speed_threshold:
            if max_course_diff <= course_threshold:
                plt.plot(batch[:,5], batch[:,6], c='black')
                plt.title('Index: %i' %index)
                plt.show()
                
            else:
                plt.plot(batch[:,5], batch[:,6], c='red')
                plt.title('Index: %i' %index)
                plt.show()
     
    # delete everything from batches according to del_indices
    result = [i for j, i in enumerate(batches) if j not in del_indices]
    print(n_omissions)
    return result, del_indices
    

batches = module.x_batches[0:120]

outlier = module.x_batches[121]
plt.plot(outlier[:,5], outlier[:,6], c='red')

straight = module.x_batches[50]
plt.plot(straight[:,5], straight[:,6], c='red')

batch = batches[0]
plt.plot(batch[:,5], batch[:,6], c='red')


result, del_indices = isolate_fishing_behaviour(batches, True, 5, 6)



# =============================================================================
# batches = module.x_batches[15:16]
# =============================================================================
# =============================================================================
# batches = module.x_batches[119:120]
# =============================================================================
# =============================================================================
# omit using standard deviation
# =============================================================================
# =============================================================================
# n_omissions = 0
# course_change_threshold = 80
# for batch in batches:
#     batch_course = batch[:, 4]
#     std_course = np.std(batch_course)
#     if (std_course >= course_change_threshold):   
#         plt.plot(batch[:,5], batch[:,6], c='red')
#         plt.show()
#     else:
#         n_omissions = n_omissions + 1
#         plt.plot(batch[:,5], batch[:,6], c='black')
#         plt.show()
#         
# print(n_omissions)
# =============================================================================
    



# =============================================================================
# find std within batch (for course change) and make speed less than 8 knots
# =============================================================================






















# =============================================================================
# visualise interpolated and raw data trajectories
# =============================================================================

threshold = 50
t2 = 5000

array_np = module.interpolated_vessels[0][20000:20000 + threshold, :]
# =============================================================================
# plt.scatter(array_np[:,5], array_np[:,6], s=1, c='black')
# =============================================================================
plt.plot(array_np[:,5], array_np[:,6])
plt.show()

df1 = module.list_df_resampled[0].tail(threshold)
# =============================================================================
# plt.scatter(df1['lat'], df1['lon'], s=1, c='black')
# =============================================================================
plt.plot(df1['lat'], df1['lon'], c='green')
plt.show()

df2 = module.list_df[0].head(t2)
# =============================================================================
# plt.scatter(df2_1['lat'], df2_1['lon'], s=1, c='black')
# =============================================================================
plt.plot(df2['lat'], df2['lon'], c='red')
plt.show()


# =============================================================================
# comparison of mmsi's
# =============================================================================


# =============================================================================
# find standard deviation of course to isolate streaming behaviour (non fishing behaviour)
# =============================================================================






# =============================================================================
# pattern vis using class pattern_visualistaion
# =============================================================================

df = pd.read_csv('../data/' + dataset + '.csv')
df_ppm = module.list_df[0]
from pattern_visualisation import segment_dataframe, random_vis
list_df = segment_dataframe(dataframe=df, time_period=24, threshold=48)


sub_df_original = df.head(15)
plt.plot(sub_df_original['lat'], sub_df_original['lon'])


sub_df = df_ppm.head(500)
plt.plot(sub_df['lat'], sub_df['lon'])


