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
Problem fixed and implemented in pre_processing_module_v2.py and combine_datasets_v2.py, results are more trustworthy
Verified the continuity and integrity of data by visualising the lat and lon in the Dataloader class (coherent trajectories are present)


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
	- Gantt chart for project timeline
	- MVP completed
	- Core technologies added
	

Mon Jan 16 17:02:14 2023
------------------------------------------------>
Linear interpolation completed for frequency of 5 minutes
Models trained on the data (CNN and GRU)
Memory limitations on 1 minute frequency

Tue Jan 24 11:41:37 2023
------------------------------------------------>
More memory obtained (from 32GB to 80GB)
1 minute frequency linearly interpolated data processed successfully
    
    
Wed Feb 1 19:10:58 2023
------------------------------------------------>
Work commenced on mTAN conversion for AIS data
Hit roadblocks with CUDA errors
Debugging the human activity data prepreprocessing
Accessed data structure for classification encoder mTAN


Fri Feb 10 22:32:14 2023
------------------------------------------------>
Organised paper for submission to KES and MAKE 2023 conferences (a lot of LaTeX work...)

Fri Feb 17 18:05:19 2023
------------------------------------------------>
Encountered error in GRU forward pass where softmaxing using cross entropy loss (incorrect as CEL should take logits not probs from softmax)
Removed softmaxing and tried two different techniques for accessing final hidden states of GRU
Accuracy is the same at 95% for the GRU, updated tables and structures for results


Sun Feb 26 15:53:04 2023
------------------------------------------------>
Tweeks to the classification and GRU layers of the mTAN network
Running experiments on mTAN network (very slow)
Researched techniques for SVM multiclass classification using multivariate time series data


Mon Mar 6 16:13:09 2023
------------------------------------------------>
Tuning of mTAN is ongoing due to long training times, best val acc so far is 92% -> update: 93%
Working on how to process data for SVM or RF
Feature extraction is a strong possibility as shown by Sanchez Pedroche et al.
Began feature extraction, computationally expensive
Added preprocessing for feature extraction to combine_datasets.py
Began layout for part 3 LaTeX


Wed Mar 15 15:09:32 2023
------------------------------------------------>
After much trial and error with feature extraction and other methods for implementing the SVM, the dataset is simply too large for this model. 
Therefore, a new tactic to implement a traditional machine learning classification model has been undertaken: logistic regression with PyTorch
Simultaneously, final tests need to be conducted on the attention network for GRU and CNN classification modules to produce results
Introduce AUC as metric for write up?

Thu Mar 16 08:58:15 2023
------------------------------------------------>
How does a linear model take the time series data?

    
BACKLOG:
------------------------------------------------>
In progress:
* Train 1D CNN and GRU models on linearly interpolated data using similar methodology and complexity, gain best possible results - done for lower resolution
* Non-linear interpolation using the reference time points generated on the AIS data then classify - in progress
* Implement SVM on all data pipelines for comparison - research in progress
	+ Problem encountered - data structure is wrong format for trad ML models
	+ SVM computation is far too expensive due to the number of examples, an alternative data processing method has to be implemented
* Try feature extraction, seems to be the commonly done thing - Sanchez Pedroche et al. and Kim and Lee use this to test SVMs and DTs
* Keep tuning mTAN, try embed_time up to a much larger number - use nonrandom init

Not started:
* Begin evaluation of results
* Introduce time measurement during prediction to see the efficiency of performance -> important for deployment
* Research applications for real time data

Finished:
* Change graphs for model performance to CNN on left not right - done
* Create reference time point set for both interpolation techniques - done (not needed in the end)
* Create linearly interpolated data with higher resolution (every minute), using custom PyTorch Dataset class for memory management - done
* Implement mTANs on human activity (MIMIC III access impossible) dataset for testing - done





==============================================================================
Write up notes:
==============================================================================
Concepts:
Explore attention mechanism, time embeddings (positional vs learned)
Logistic regression -> linear model for comparison

Metrics:
Introduce AUC as metric and disadvantages for class imbalance?
Or maybe average precision to determine the performance on smaller classes?
Introduce time per epoch as metric for evaluation

Results:
Results of lin interp CNN and GRU
Results from mTAN CNN and GRU
Results from feature extraction to ML model (linear and svm perhaps?) - in doubt

Statistical analysis:
avg distance (distance of each sequence) per class
avg speed per class
avg cumulative course change per class
visualisations!

Further discussion:
Remember all challenges faces:   
    
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

	1) Create time point vector (cumulative sum of time deltas)
	2) Feed into mTAN network and attack classification network, no need to decode when the embeddings can be fed straight into the classification network

	    
"""

