#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:18:55 2022

@author: benedict

Script implementing customised PyTorch Dataset and DataLoader classes

"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import pickle

import matplotlib.pyplot as plt

class AIS_loader(Dataset):
    
# =============================================================================
#     structure of input: list of sequences (shuffled already)
#     each sequence is: sequence length x features
#     feature columns: speed, lat, lon, time_delta, course_delta
# =============================================================================
    
    
    def __init__(self, choice, split, version, split_2=None):
        random.seed(15) # set random seed to reproduce random results
        with open(f'../../data/pkl/{choice}/{split}_v{version}.pkl', 'rb') as f:
            self.seq_list = pickle.load(f)
        if split_2 != None:
            with open(f'../../data/pkl/{choice}/{split_2}_v{version}.pkl', 'rb') as f:
                self.seq_list += pickle.load(f)
        with open(f'../../data/pkl/{choice}/utils_v{version}.pkl', 'rb') as f:
            obj = pickle.load(f)
            self.n_features = obj[0]
            self.n_classes = obj[1]
            self.seq_length = obj[2]
            
                
    def __getitem__(self, index):
        sequence = self.seq_list[index]
        features = sequence[:, :-1]
        labels = sequence[0, -1]
        return torch.tensor(features), labels, len(features)
        
    
    def __len__(self):
        return len(self.seq_list)
    
    
    # =============================================================================
    # takes output of Dataset class (AIS_loader) __getitem__ which is a tuple of (tensor, labels, lengths)
    # converts the irregularly sized sequences into the same lengths (taking the max length of all the sequences)
    # =============================================================================
    def CNN_collate(self, data):
        
        seqs, labels, lengths = zip(*data)
        max_len = self.seq_length
        n_features = 5
    
        sequences = torch.zeros(len(data), max_len, n_features) # create empty padded tensor to fill in loop below
        labels = torch.tensor(labels)
        lengths = torch.tensor(lengths)
        
        for i in range(len(data)):
            j, k = data[i][0].size(0), data[i][0].size(1)
            sequences[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])
        
        sequences = torch.transpose(input=sequences, dim0=1, dim1=2) # transpose sequences
        
        return sequences.float(), labels.long(), lengths.long()
    
    
    def GRU_collate(self, data):
        seqs, labels, lengths = zip(*data)
        max_len = self.seq_length
        n_features = 5
    
        sequences = torch.zeros(len(data), max_len, n_features) # create empty padded tensor to fill in loop below
        labels = torch.tensor(labels)
        lengths = torch.tensor(lengths)
        
        for i in range(len(data)):
            j, k = data[i][0].size(0), data[i][0].size(1)
            sequences[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])
        
        
        return sequences.float(), labels.long(), lengths.long()

        
    # =============================================================================
    # method that randomly visualises the tensor sequences from the class
    # =============================================================================
    def visualise_tensor(self, n_iters):
        for i in range(n_iters):
            random_int = random.randint(0, len(self.seq_list))
            feature, label, length = self.__getitem__(random_int)
            plt.title(f'index: {random_int}, label: {label}, seq_length: {length}, tensor')
            plt.plot(feature[:, 1], feature[:, 2])
            plt.scatter(feature[:, 1], feature[:, 2], s=8)
            plt.show()


# =============================================================================
# testing zone
# =============================================================================

# dataset = AIS_loader(choice='varying', split='train', version=3)
# dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, collate_fn=dataset.GRU_collate)

# for features, labels, lengths in dataloader:
#     print(labels.size())
# dataset.visualise_tensor(100)


# seq_list = dataset.seq_list
# single = seq_list[0]
# single = seq_list[14705]
# plt.plot(single[:, 1], single[:, 2])
# plt.scatter(single[:, 1], single[:, 2], s=8)










