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
Created the 1D_CNN PyTorch model
    - Extensive problems with exploding gradient caused by NaN values in the linearly interpolated data, solved by re-normalising the aggregated data


Sat Nov 26 12:11:21 2022
------------------------------------------------>
Rebuilt data pipeline due to memory constraints with the load_data class.
	- Implemented the PyTorch Dataset class (AIS_loader.py), introduced a custom collate_fn to pad the tensors with zeros
Fine tuning CNN class, using nonrandom parameter initialisation according to a Reed and Marks technique to explore the multimodal error landscape and find the best starting parameters to then do more extensive training, starting with those parameters.
Novel additiion to the Reed and Marks nonrandom initialisation by using validation accuracy as the defining metric as opposed to training loss, this allows the exploration to focus on generalisation as a priority


Mon Nov 28 10:43:50 2022
------------------------------------------------>
Created padding function in AIS_loader class (using tensors not numpy arrays)
	- This has allowed batching for training, massively increasing throughput and training efficiency. 
Created the 1D_CNN_v1.py class inheriting from PyTorch nn.Module class
Created wrapper class for the 1D_CNN_v1.py class to introduce fit, predict, confusion_matrix, visualisation and more methods and class attributes for more efficient tracking of results and training


Wed Nov 30 15:27:35 2022
------------------------------------------------>
Suspiciously accurate results for the 1D_CNN (98%), investigating the cause - nothing so far
    
    
Sat Dec 3 17:19:54 2022
------------------------------------------------>
Error found in the data processing, specifically combine_datasets.py, led to jumbled sequences causing the false positives
Problem fixed and implemented in pre_processing_module_v2.py and combine_datasets_v2.py, results are more believable


Mon Dec 5 18:09:14 2022
------------------------------------------------>
Results from the 1D_CNN stalled at accuracy 88%, introduced residual blocks to help with stability and performance as the network complexity grows
Results already up to 91% with res blocks, continuing to tweak hyperparams to get better results


Tue Dec 6 22:03:48 2022
------------------------------------------------>
Res block 1D CNN best results so far with more than 93% test accuracy after validation and training results were combined for a final run. Strong results
Implemented GRU, starting with 1 GRU module, no droupout and mono directional. Results are promising and with further tweaking results could surpass res block CNN


Wed Dec 7 19:29:36 2022
------------------------------------------------>
Stacked, bidirectional GRU with dropout and 128 hidden units is performing excellently on the padded, featurised data with final results of 95% test accuracy, when utilising custom non-random initialisation technique
Very stable training, clearly a strong model for time series data, even irregular data 

Thu Dec 8 21:15:27 2022
------------------------------------------------>
Introduced L1-norm global unstructured weight pruning to push the results of both the GRU and CNN to higher accuracies on test data


Fri Dec 9 21:51:29 2022
------------------------------------------------>
Researched GIS software for visualising the trajectories, ArcGIS seemed like the best prospect but was paywalled. Therefore Q-GIS was implemented and used for the data visualisations
Captured figures for selected trajectories of each fishing class for section 3 of the research paper


Mon Dec 12 13:07:59 2022
------------------------------------------------>
Prototype results gathered for design specification hand in, write up beginning now. 


Mon Dec 19 11:26:21 2022
------------------------------------------------>
Completed Preliminary results section
Tables for each GRU and CNN
	- Structure table
	- Hyperparameters table
	- Results table
Figures for GRU and CNN

	
Tue Dec 20 15:52:47 2022
------------------------------------------------>
Organised tables
GRU and CNN equations added
Completed references
Introduced abstract
Appendix:
	Gantt chart for project timeline
	MVP completed
	Core technologies added
	

    
    
    
    
    
BACKLOG:
------------------------------------------------>
* Change graphs for model performance to CNN on left not right
* Create reference time point set for both interpolation techniques
* Create linearly interpolated data with higher resolution (every minute), using custom PyTorch Dataset class for memory management
* Train 1D CNN and GRU models on linearly interpolated data using similar methodology and complexity, gain best possible results
* Implement mTANs on human activity (MIMIC III access impossible) dataset for testing
* Non-linear interpolation using the reference time points generated on the AIS data
* Train classification networks on the non-linearly interpolated data and gain best possible results
* Implement SVM on all data pipelines for comparison
* Begin evaluation of results
* Introduce time measurement during prediction to see the efficiency of performance -> important for deployment
* Research applications for real time data
   
    
DATA PIPELINES:
------------------------------------------------>
Collated and padded varying sequence lengths:

	1) Script combine_datasets.py loads raw csv data using the pre_processing_module.py, processes it into sequences of 24hrs, adds labels and combines into an aggregate dataset. 
		Two options for combine_datasets.py:
			1) Process the raw data from csv and save as parquet
			2) Just load from previously saved parquet and do final processing
	2) In combine_datasets.py once the data has had prelim processing and is aggregated, the sequences are shuffled, split, flattened then normalised before re-segmenting into original sequences using a mask. This process, although rather convoluted maintains the sequences, while splitting and normalising the data fairly i.e. maintains anonymity of test and validation data for the model training
	3) Implementing customised PyTorch classes Dataset (AIS_loader.py) and DataLoader is preserving memory for the padded sequences as using load_data class was consuming too much of the GPU VRAM.
	4) Although the padding is extensive, the models are training efficiently on the data
	


Linear interpolation:

	1) Create reference time points
	2) Merge irregular time series data and the reference time points
	3) Interpolate using formula provided
	4) Remove original data points, leaving only the evenly sampled time series data

mTAN non-linear interpolation network:

	1) Create reference time points
	2) ?
	    
"""

