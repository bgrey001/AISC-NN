#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:20:48 2022

@author: benedict

Script used for loading data into the network script and converting sequnces to tensors
"""


import pickle
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random

class data_loader():
    
    
    # =============================================================================
    # constructor
    # =============================================================================
    def __init__(self, choice, version):
        
        self.choice = choice
        self.version = version
        
        random.seed(15) # set random seed to reproduce random results
        with open(f'../../data/pkl/{choice}/train_v{version}.pkl', 'rb') as f:
            self.train_seq_list = pickle.load(f)
        with open(f'../../data/pkl/{choice}/valid_v{version}.pkl', 'rb') as f:
            self.valid_seq_list = pickle.load(f)
        with open(f'../../data/pkl/{choice}/test_v{version}.pkl', 'rb') as f:
            self.test_seq_list = pickle.load(f)
        if choice == 'varying': # temporary measure to prevent error
            with open(f'../../data/pkl/{choice}/utils_v{version}.pkl', 'rb') as f:
                obj = pickle.load(f)
                self.n_features = obj[0]
                self.n_classes = obj[1]
                self.seq_length = obj[2]

                
    # =============================================================================
    # shuffle the sequences for the network
    # =============================================================================
    def load_shuffled(self):
        random.shuffle(self.train_seq_list)
        random.shuffle(self.valid_seq_list)
        random.shuffle(self.test_seq_list)
        # return self.train_seq_list, self.valid_seq_list, self.test_seq_list
        return self.train_seq_list, self.valid_seq_list, self.test_seq_list
    
    # =============================================================================
    # takes a list of data e.g train_seq_list and returns it in batch sizes
    # =============================================================================
    def load_batch_shuffled(self, data_list, batch_size):
       return [data_list[i * batch_size:(i + 1) * batch_size] for i in range((len(data_list) + batch_size - 1) // batch_size )][:-1]  
   
    
   
    # =============================================================================
    # shuffle the sequences in a given list
    # =============================================================================
    def shuffle_data(self, data):
        random.shuffle(data)
        return data
    
    
    # =============================================================================
    # load the data as it is, unshuffled
    # =============================================================================
    def load_unshuffled(self):
        return self.train_seq_list, self.valid_seq_list, self.test_seq_list
    
    
    
    # =============================================================================
    # takes a sequence from a sequence_list and returns an input_tensor and target tensor ready for the GRU
    # =============================================================================
    def seq_to_tensor(self, seq): 
        # n_features = 4 
        # shape is timesteps, 1, n_features
        input_tensor = torch.zeros(len(seq), 1, self.n_features)
        target_tensor = torch.zeros(1)
        for i, item in enumerate(seq):
            input_tensor[i][0][:] = torch.tensor(item[:-1])
            target_tensor[0] = torch.tensor(item[-1])
            
        return input_tensor, target_tensor


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
    # input is currently linearly interpolated, evenly spaced sequences
    # =============================================================================
    def batch_seq(self, batch):
        # tensor dimension: batch_length, n_features, seq_length        
        match self.choice:
            case 'varying':
                # convert batch of arrays to batch of tensors
                for i, item in enumerate(batch):
                    batch[i] = torch.tensor(item)
                # now pad the batches
                for i, item in enumerate(batch):
                    batch[i] = F.pad(item, pad=(0, 0, 0, self.seq_length - item.size()[0])) # sequence length is self.seq_length
                tensor = torch.stack(batch) # stack the list of tensors into a tensor
                tensor = torch.transpose(input=tensor, dim0=1, dim1=2)
                input_tensor = tensor[:, :-1, :]
                target_tensor = tensor[:, -1, :]
                target_tensor = target_tensor[:, -1]
                return input_tensor, target_tensor
            
            case 'linear_interp':
                tensor = torch.tensor(batch)
                tensor = torch.transpose(input=tensor, dim0=1, dim1=2)
                input_tensor = tensor[:, :-1, :]
                target_tensor = tensor[:, -1, :]
                target_tensor = target_tensor[:, -1]
                return input_tensor, target_tensor




    def return_classes(self):
        return ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers']





dl = data_loader(choice='varying', version='2')
# print(dl.n_features)
seq_list = dl.train_seq_list
# batches = dl.load_batch_shuffled(data_list=seq_list, batch_size=256)
# batch = batches[0]

        # train_batches = self.data_loader.load_batch_shuffled(data_list=self.train_data, batch_size=self.batch_size)

# b = torch.tensor(batch[0])
# t = F.pad(b, pad=(0, 0, 0, 500 - b.size()[0]))
# tensor = torch.transpose(input=t, dim0=0, dim1=1)
# b3 = tensor.numpy()

# b1 = t.numpy()

# x, y = dl.batch_seq(batch)
# # batch = batches[0]


# x, y = dl.batch_seq(batch)

# # print(x[0])

# seq = batch[0]
# seq_len = 2941


# for i, item in enumerate(batch):
#     batch[i] = torch.tensor(item)

    
    
# batch_tensors = pad_sequence(batch, batch_first=True)
# print(batch_tensors.size())

# t = batch[0]

# print(t.size()[0])

# for i, item in enumerate(batch):
#     batch[i] = F.pad(item, pad=(0, 0, 0, seq_len - item.size()[0]))    
#     # 
    
    
# padded_seq = F.pad(torch.tensor(seq), pad=(0, 0, 0, seq_len - )
# padded_batch = F.pad(torch.tensor(batch), pad=(0, 0, 0, seq_len - len(seq)))

# dl.batch_padding(batch)



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


