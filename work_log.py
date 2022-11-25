#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:03:16 2022

@author: BenedictGrey

WORK LOG

Mon Oct 15 17:03:16 2022
------------------------------------------------
Created the following files:
    - trollers_preprocessing.py
    - trawlers_preprocessing.py
    - purse_seines_preprocessing.py
    - pole_and_line_preprocessing.py
    

Sun Nov 13 12:01:36 2022
------------------------------------------------
Created the following files:
    - pre_processing module
    	- extensive class containing many of the necessary custom data processing functions
    	- creates time windowed data, linearly interpolated, padded and varying datasets
    	- implements train, test split from sklearn to create partition data
    - combine_datasets module
    	- module that combines all the seperate datasets into one and normalises
    - load_data module
    
    
    
Mon Nov 21 19:16:43 2022
------------------------------------------------
Created GRU-RNN PyTorch model for varying length sequences
    - Trained on v1, v2, v3 of the varying dataset 	
    - Results on the first 3 versions of data are void due to sampling error
    	- Samping error was creating even distribution across the classes
Modified pre_processing module to create natural class sample distribution
    - Further modifications include introducing linear interpolation, necessary for training the CNN
Created the 1DCNN PyTorch model
    - Extensive problems with exploding gradient caused by NaN values in the linearly interpolated data, solved by re normalising the aggregated data
    - 

    
TODO:
------------------------------------------------
    
   
    
    
    
   
   
   
   
Notes:

How to figure out number of neurons needed in the fully connected layer

Layer 1
Each input is 4 features, 180 sequence length (8 minute intervals)
4 channels in (for each feature)
maxpool is 2

output1 = out_channels1 x seq_length - (kernel_size - 1)
	= 16 x 180 - (2 - 1)
	= 16 x 179 	
divide dim=1 by maxpoolsize and take floor -> floor(179/2) = 89 
	= 16 x 89

layer 2
output2 = out_channels2 x output1(dim=1) - (kernel_size - 1)
	= 32 x 89 - (2 - 1)
	= 32 x 88
divide again by pool size
	= 32 x (floor(88/2))
	= 32 x 44
	
layer 3 
output3 = out_channels3 x output2(dim=1) - (kernel_size - 1)
	= 64 x 43 
divide again
	= 64 x floor(43/2)
	= 64 x 21
    
    
"""

