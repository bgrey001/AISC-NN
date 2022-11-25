#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:20:48 2022

@author: benedict

Script used for loading data into the network script and converting sequnces to tensors
"""


import pickle
import torch
import numpy as np
import random

class data_loader():
    
    
    # @classmethod
    # =============================================================================
    # constructor
    # =============================================================================
    def __init__(self, choice, version):
        
        random.seed(15) # set random seed to reproduce random results
        with open(f'../../data/pkl/{choice}/train_v{version}.pkl', 'rb') as f:
            self.train_seq_list = pickle.load(f)
            
        with open(f'../../data/pkl/{choice}/valid_v{version}.pkl', 'rb') as f:
            self.valid_seq_list = pickle.load(f)
        
        with open(f'../../data/pkl/{choice}/test_v{version}.pkl', 'rb') as f:
            self.test_seq_list = pickle.load(f)
            
    # @classmethod
    # =============================================================================
    # shuffle the sequences for the network
    # =============================================================================
    def load_shuffled(self):
        random.shuffle(self.train_seq_list)
        random.shuffle(self.valid_seq_list)
        random.shuffle(self.test_seq_list)
        # return self.train_seq_list, self.valid_seq_list, self.test_seq_list
        return self.train_seq_list, self.valid_seq_list, self.test_seq_list
    
    # @classmethod
    # =============================================================================
    # takes a list of data e.g train_seq_list and returns it in batch sizes
    # =============================================================================
    def load_batch_shuffled(self, data_list, batch_size):
       return [data_list[i * batch_size:(i + 1) * batch_size] for i in range((len(data_list) + batch_size - 1) // batch_size )][:-1]  
   
    
   
    # @classmethod
    # =============================================================================
    # shuffle the sequences in a given list
    # =============================================================================
    def shuffle_data(self, data):
        random.shuffle(data)
        return data
    
    
    # @classmethod
    # =============================================================================
    # load the data as it is, unshuffled
    # =============================================================================
    def load_unshuffled(self):
        return self.train_seq_list, self.valid_seq_list, self.test_seq_list
    
    
    
    
    # @classmethod
    # =============================================================================
    # takes a sequence from a sequence_list and returns an input_tensor and target tensor ready for the GRU
    # =============================================================================
    def seq_to_tensor(self, seq): 
        n_features = 4 
        # shape is timesteps, 1, n_features
        input_tensor = torch.zeros(len(seq), 1, n_features)
        target_tensor = torch.zeros(1)
        for i, item in enumerate(seq):
            input_tensor[i][0][:] = torch.tensor(item[:-1])
            target_tensor[0] = torch.tensor(item[-1])
            
        return input_tensor, target_tensor




    # @classmethod
    # =============================================================================
    # takes a sequence from a sequence_list and returns an input_tensor and target tensor ready for the CNN
    # =============================================================================
    def cnn_seq(self, seq):
        # n_features = 4

        target_tensor = torch.tensor(seq[0, -1]).unsqueeze(0)
        input_tensor = torch.tensor(seq[:, :-1]).transpose(0, 1).unsqueeze(0)

        # print(target_tensor)
        return input_tensor, target_tensor
    
    
    
    # =============================================================================
    # takes a batch and returns the tensors with all the inputs and targets in that batch
    # =============================================================================
    def batch_cnn_seq(self, batch):
        
        # labels = np.array(batch)
        # labels = torch.tensor(labels[:, :, -1])
        
        tensor = torch.tensor(batch)
        tensor = torch.transpose(input=tensor, dim0=1, dim1=2)
        input_tensor = tensor[:, :-1, :]
        target_tensor = tensor[:, -1, :]
        target_tensor = target_tensor[:, -1]
        
        return input_tensor, target_tensor
        



    def return_classes(self):
        return ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers']





# dl = data_loader(choice='linear_interp', version='2')
# all_batches = dl.load_batch_shuffled(dl.valid_seq_list, 128)
# single_batch = all_batches[5]

# labels = np.array(single_batch)
# labels = labels[:, :, -1]

# i, t, l = dl.batch_cnn_seq(single_batch)


# t1 = t[:, -1]
# print(t1)
# single_batch = all_batches[0]

# # for seq in single_batch:
    
# tensor = torch.tensor(single_batch)
# tensor1 = torch.transpose(input=tensor, dim0=1, dim1=2)
# target = tensor1[:, -1, :]
# print(target)

# input = tensor1[:, :-1, :]

# lst1 = tensor1.numpy()
# print(len(lst1[0]))

# data_train, data_valid, data_test = dl.load_shuffled()


