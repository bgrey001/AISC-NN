#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:03:16 2022

@author: BenedictGrey

WORK LOG

Mon Oct 15 17:03:16 2022
------------------------------------------------>
Created the following files:
    - trollers_preprocessing.py
    - trawlers_preprocessing.py
    - purse_seines_preprocessing.py
    - pole_and_line_preprocessing.py
    

Sun Nov 13 12:01:36 2022
------------------------------------------------>
Created the following files:
    - pre_processing module
    	- extensive class containing many of the necessary custom data processing functions
    	- creates time windowed data, linearly interpolated, padded and varying datasets
    	- implements train, test split from sklearn to create partition data
    - combine_datasets module
    	- module that combines all the seperate datasets into one and normalises
    - load_data module
    
    
    
Mon Nov 21 19:16:43 2022
------------------------------------------------>
Created GRU-RNN PyTorch model for varying length sequences
    - Trained on v1, v2, v3 of the varying dataset 	
    - Results on the first 3 versions of data are void due to sampling error
    	- Samping error was creating even distribution across the classes
Modified pre_processing module to create natural class sample distribution
    - Further modifications include introducing linear interpolation, necessary for training the CNN
Created the 1DCNN PyTorch model
    - Extensive problems with exploding gradient caused by NaN values in the linearly interpolated data, solved by re normalising the aggregated data



Sat Nov 26 12:11:21 2022
------------------------------------------------>
Rebuilt data pipeline due to memory constraints with the load_data class.
	- Implemented the PyTorch Dataset class (AIS_loader.py), introduced a custom collate_fn to pad the tensors with zeros
Fine tuning CNN class, using a K random parameter initialisation technique to explore the multi model error landscape and find the best starting parameters to then do more extensive training, starting with those parameters



    
BACKLOG:
------------------------------------------------>
1) Create padding function in load data (using tensors not numpy arrays) -> finished
4) Train CNN and GRU on padded data -> in progress


2) Map trajectories onto a world map using cartopy fot the data section (section 3) of the research paper
3) Create linearly interpolated data with higher resolution (every 5 minutes), using custom PyTorch Dataset class
5) Train and test CNN on new padded data and linearly interpolated data
6) Build and train GRU on padded and linearly interpolated data
7) Implement mTANs on Mimic III dataset for testing
   
   
   
   
   
   
   
    
DATA PIPELINES:
------------------------------------------------>
Collated and padded varying sequence lengths:

	1) Script combine_datasets.py loads raw csv data using the pre_processing_module.py, processes it into sequences of 24hrs, adds labels and combines into an aggregate dataset. 
		Two options for combine_datasets.py:
			1) Process the raw data from csv and save as parquet
			2) Just load from previously saved parquet and do final processing
	2) In combine_datasets.py once the data has had prelim processing and is aggregated, the sequences are shuffled, split, flattened and then 			normalised before re-segmenting into original sequences using a mask. This process, although rather convoluted maintains the sequences, while splitting and normalising the data fairly i.e. maintains anonymity of test and validation data for the model training
	3) Implementing customised PyTorch classes Dataset (AIS_loader.py) and DataLoader is preserving memory for the padded sequences as using load_data class was consuming too much of the GPU VRAM.
	4) Although the padding is extensive, the models are training efficiently on the data
	


Linear interpolation:

	1)

	
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
Tim notes:
Framework that we can build on

Problems overcame in data

Keep style

Breakdown of the system

Models building

use tables for results

Problem solving

Abstract at the start
Present preliminary results

More exotic?

Non neural network modelling
Best result and comparison
Don't explain, just reference






Appendix:
Risk analysis
Project planb
Have I achieve the mielstones?


	    
	    
	    
	    
	    
	    
	    
	    
	   
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

