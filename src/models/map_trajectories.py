#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 15:48:39 2022

@author: BenedictGrey

Script to map the trajectories of the varying sequences (uninterpolated to a real world map using cartopy)
"""


import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs





    
single = seq_list[0]

    

def plot_sequence(seq):
    plt.plot(seq[:, 2], seq[:, 3])
    plt.show()
    
    
plot_sequence(single)