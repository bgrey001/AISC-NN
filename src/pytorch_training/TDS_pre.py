#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:20:29 2022

@author: benedict

TDS tutorial, data preprocessing


"""
# =============================================================================
# import
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data/PJME_hourly.csv')
df = df.set_index(['Datetime'])
df.index = pd.to_datetime(df.index)

if not df.index.is_monotonic_increasing: # sort so the order is ascending
    df = df.sort_index()
    
df = df.rename(columns={'PJME_MW' : 'value'})

# =============================================================================
# plotting, create a mask for date range - can possibly use for project
# =============================================================================
start_date = datetime.datetime(2012, 1, 1)
end_date = datetime.datetime(2012, 6, 1)
mask = (df.index > start_date) & (df.index <= end_date)

df_plt = df.loc[mask]
# df_plt = df[df.index >= start_year]
plt.figure(figsize=(20, 3))
plt.plot(df_plt.index[:], df_plt.value)
        

# =============================================================================
# generate features to create multivariate data
# =============================================================================

# creating lagged observations as features
def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n['value'].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n

input_dim = 100 # input dim for the network is going to be the number of features, which is what we are creating with generate_time_lags
df_generated = generate_time_lags(df, input_dim)


# generating features from timestamps
df_features = (
                df
                .assign(hour = df.index.hour)
                .assign(day = df.index.day)
                .assign(month = df.index.month)
                .assign(day_of_week = df.index.dayofweek)
                .assign(week_of_year = df.index.week)
                )


# one way to encode datatime features is to treat them as categorical variables using one hot encoding
def onehot_encode(df, onehot_columns):
    ct = ColumnTransformer(
        [('onehot', OneHotEncoder(drop='first'), onehot_columns)],
        remainder='passthrough'
        
        )
    return ct.fit_transform(df)

onehot_columns = ['hour','month','day','day_of_week','week_of_year']
onehot_encoded = onehot_encode(df_features, onehot_columns)
df_new = pd.DataFrame.sparse.from_spmatrix(onehot_encoded)




# pandas onehot encoding
def onehot_encode_pd(df, col_name):
    dummies = pd.get_dummies(df[col_name], prefix=col_name)
    return pd.concat([df, dummies], axis=1).drop(columns=[col_name])

# df_features = onehot_encode_pd(df_features, ['week_of_year'])


# generating cyclical time features 

def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
             }
    return df.assign(**kwargs).drop(columns=[col_name])

df_features = generate_cyclical_features(df_features, 'hour', 24, 0)
df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)
df_features = generate_cyclical_features(df_features, 'month', 12, 1)
df_features = generate_cyclical_features(df_features, 'week_of_year', 52, 0)

# holidays
from datetime import date
import holidays

us_holidays = holidays.US()

def is_holiday(date):
    date = date.replace(hour = 0)
    return 1 if (date in us_holidays) else 0

def add_holiday_col(df, holidays):
    return df.assign(is_holiday = df.index.to_series().apply(is_holiday))

df_features = add_holiday_col(df_features, us_holidays)

# =============================================================================
# splitting data into train, validation and test sets
# =============================================================================

from sklearn.model_selection import train_test_split

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio/ (1 - test_ratio)










