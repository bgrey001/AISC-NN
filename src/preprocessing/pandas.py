#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:58:03 2023

@author: Benedict Grey
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

r = np.random.normal(0,1, 100)
plt.plot.hist(r)
# =============================================================================
# awful groupby one
# =============================================================================
n = 500
df = pd.DataFrame(zip(list(range(n)), np.random.randint(9,12+1,n), np.random.randint(0,100+1,n)), columns=['user_id', 'grade', 'test score']).set_index('user_id')

bin_labels = ['<50', '<75', '<90', '<100']
bins = [0, 50, 75, 90, 100]
df['test score'] = pd.cut(df['test score'], bins=bins, labels=bin_labels)

df_grouped = df.groupby(['grade', 'test score'], as_index=False).size()
df_grouped['cumsum'] = df_grouped.groupby('grade').cumsum()
max_per_group = df_grouped.groupby('grade')['cumsum'].max()
df_grouped['max_per_group'] = df_grouped['grade'].map(max_per_group) 
df_grouped['percentage'] = (df_grouped['cumsum'] / df_grouped['max_per_group'] * 100).astype(int).astype(str) + "%"
final_df = df_grouped[['grade', 'test score', 'percentage']].set_index('grade')
df_grouped2 = df.groupby('grade')['test score'].size()





def buck_test_score(df):
    # make labels for bins and bin values, essentially what are the intervals to bucket each value of test score
    bin_labels = ['<50', '<75', '<90', '<100']
    bins = [0, 50, 75, 90, 100]
    # create buckets of test_scores
    df['test score'] = pd.cut(df['test score'], bins=bins, labels=bin_labels)
    # count the number of people in each bucket by grades
    df_aux = df.groupby(['grade', 'test score'], as_index=False).size()
    # get cumalative sum of each grades buckets
    df_aux['cumsum'] = df_aux.groupby('grade')['size'].cumsum()
    # find max of each group
    maxima = df_aux.groupby('grade')['cumsum'].max()
    # add max value within group to each row within group
    df_aux['max_per_group'] = df_aux['grade'].map(maxima)
    # find percentage and convert to string
    df_aux['percentage'] = ((df_aux['cumsum'] / df_aux['max_per_group'] * 100).astype(int)).astype(str) + "%"
    # df to return
    ret_df = df_aux[['grade', 'test score', 'percentage']]
    return ret_df
    
    
    
# =============================================================================
# combine strings
# =============================================================================

# addresses = {"address": ["4860 Sunset Boulevard, San Francisco, 94105", "3055 Paradise Lane, Salt Lake City, 84103", "682 Main Street, Detroit, 48204", "9001 Cascade Road, Kansas City, 64102", "5853 Leon Street, Tampa, 33605"]}
# cities = {"city": ["Salt Lake City", "Kansas City", "Detroit", "Tampa", "San Francisco"], "state": ["Utah", "Missouri", "Michigan", "Florida", "California"]}

# df_addresses = pd.DataFrame(addresses)
# df_cities = pd.DataFrame(cities)

def combine_address(df_addresses: pd.DataFrame, df_cities: pd.DataFrame):
    ret_df = pd.DataFrame(columns=['address'])
    for i, row in enumerate(df_addresses['address']):
        array = row.split(', ')
        city = array[1]
        for j, city_2 in enumerate(df_cities.city): 
            if city == city_2:
                array.insert(2, df_cities.loc[j]['state'])
                ret_df.loc[len(ret_df)] = ', '.join(array)
    return ret_df
    



# =============================================================================
# good grades and favourite colours
# =============================================================================


# students = {"name" : ["Tim Voss", "Nicole Johnson", "Elsa Williams", "John James", "Catherine Jones"], "age" : [19, 20, 21, 20, 23], "favorite_color" : ["red", "yellow", "green", "blue", "green"], "grade" : [91, 95, 82, 75, 93]}
# students_df = pd.DataFrame(students)

# fave colour is green or red and grade is above 90
def conditions(students_df: pd.DataFrame):
    return students_df[((students_df['favorite_color'] == 'green') | (students_df['favorite_color'] == 'red')) & (students_df['grade'] > 90)]


# =============================================================================
# rain on rainy days
# =============================================================================
# rainfall = {"Day" : ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], "Inches" : [0, 1.2, 0, 0.8, 1]}
# df_rain = pd.DataFrame(rainfall)
def rain_median(df_rain: pd.DataFrame):
    return df_rain[df_rain['Inches'] > 0].median()





# =============================================================================
# impute median
# =============================================================================


# cheeses = {"Name": [
# "Bohemian Goat", 
# "Central Coast Bleu", 
# "Cowgirl Mozzarella", 
# "Cypress Grove Cheddar", 
# "Oakdale Colby"], 
# "Price" : [15.00, None, 30.00, None, 45.00]}

# df = pd.DataFrame(cheeses)

def impute_pandas(df: pd.DataFrame):
    # impute using median
    # first find the median of the non nan vals
    median = df['Price'].dropna().median()
    df = df['Price'].fillna(median)
    # one liner
    # df = df['Price'].fillna(df['Price'].dropna().median())
    """
    remember dropna() and fillna()
    """
    
    
    
    
    
# =============================================================================
# Weekly aggregation
# =============================================================================

# sort into list of lists by week
from datetime import datetime, timedelta

# ts = [
#     '2019-01-01', 
#     '2019-01-02',
#     '2019-01-08', 
#     '2019-02-01', 
#     '2019-02-02',
#     '2019-02-05',
# ]
def weekly_aggregation(ts: list):
    
        start_t = datetime.strptime(ts[0], "%Y-%m-%d")
        week = timedelta(days=7)
        
        sub_list = []
        ret_list = []
        
        for time in ts:
            curr_t = datetime.strptime(time, "%Y-%m-%d")
            # how far apart are the time steps from start time?
            delta = curr_t - start_t
            if delta >= week:
                start_t = curr_t
                ret_list.append(sub_list)
                sub_list = []
            # else:
            sub_list.append(str(curr_t))
        
        # now add remaining items to the sub list
        ret_list.append(sub_list)
                
                
                
# =============================================================================
# roots
# =============================================================================

# where words match with the roots, cut them into the root version

# roots = ["cat", "bat", "rat"]
# sentence = "the cattle was rattled by the battery"

def roots(roots, sentence):
    arr = sentence.split(' ')
    for i in range(len(arr)):
        for word in roots:
            if word in arr[i]:
                arr[i] = word
    ret_str = ' '.join(arr)
    return ret_str




# =============================================================================
# friends added and removed
# =============================================================================

# friends_added = [
#     {'user_ids': [1, 2], 'created_at': '2020-01-01'},
#     {'user_ids': [3, 2], 'created_at': '2020-01-02'},
#     {'user_ids': [2, 1], 'created_at': '2020-02-02'},
#     {'user_ids': [4, 1], 'created_at': '2020-02-02'}]

# friends_removed = [
#     {'user_ids': [2, 1], 'created_at': '2020-01-03'},
#     {'user_ids': [2, 3], 'created_at': '2020-01-05'},
#     {'user_ids': [1, 2], 'created_at': '2020-02-05'}]


def friend_lifecycle(friends_added, friends_removed):
    # could see if friends removed contains the sorted list of user ids from friends added
    output = []
    for added in friends_added:
        add_ids = added['user_ids']
        add_time = added['created_at']
        add_ids.sort()
        for idx, removed in enumerate(friends_removed):
            rem_ids = removed['user_ids']
            rem_time = removed['created_at']
            rem_ids.sort()
            # if there is a match we add to output dict
            if add_ids == rem_ids:
                output.append(
                    {'user_ids': add_ids,
                     'start_date': add_time,
                     'end_date': rem_time})
                del friends_removed[idx]
                break
        
    


# ==============================================================0===============
# invert matrix
# =============================================================================
# m2 = np.random.rand(5, 5)
# m2_inv = np.linalg.inv(m2)


# =============================================================================
# portion of employment
# =============================================================================
# matrix = np.array( [[10, 20, 30, 30, 10], [15, 15, 5, 10, 5], [150, 50, 100, 150, 50], [300, 200, 300, 100, 100], [1, 5, 1, 1, 2]] )

def mat_employment(matrix):
    # we want to divide each cell by the sum of the row
    matrix = np.matrix(matrix)
    ret_mat = np.zeros((5,5))
    for i in range(matrix.shape[0]):
        row_sum = np.sum(matrix[i, :])
        for j in range(matrix.shape[1]):
            print(matrix[i, j] / row_sum)
            ret_mat[i, j] = matrix[i, j] / row_sum
    
    return ret_mat


# =============================================================================
# Matrix rotation
# =============================================================================
# mat = np.array( [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]] )

# transposing is flipping on the diaganol, this is a rotation 90 degrees
# so, we can iterate in reverse through the columns and place in the rows
# i.e. first row becomes last column
# this means the rows becomes the columns and vice versa
def rotate_matrix(mat):
    n_rows = mat.shape[0]
    n_cols = mat.shape[1]
    rot_mat = np.zeros((n_cols, n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            rot_mat[j, n_cols-1-i] = mat[i, j]
    
    

# =============================================================================
# closest key
# =============================================================================
# sdct = {
#     'a' : ['b','c','e'],
#     'm' : ['c','e'],
# }
# target = 'c'

def min_dict_dist(dct, target):
    min_dict = {}
    min_idx = np.inf
        
    for i, key in enumerate(dct):
        lst = dct[key]
        for j, val in enumerate(lst):
            if (val == target) & (j < min_idx):
                min_idx = j
                min_dict = key
                break
    return min_dict

        
        
# =============================================================================
# max substring
# =============================================================================


# string2 = 'mississippi'
# string1 = 'mossyistheapple'

# string1 = 'abbc'
# string2 = 'acc'

# nested for loop, looking for matches and adding to string
def max_substring(string1, string2):
    ret_str = ''
    for i in range(len(string1)):
        for j in range(len(string2)):
            if string1[i] == string2[j]:
                ret_str += string1[i]
                break
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        






