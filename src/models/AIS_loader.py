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


class AIS_loader(Dataset):
    
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

        



# =============================================================================
# testing zone
# =============================================================================

# dataset = AIS_loader(choice='varying', split='train', version=2, split_2='valid')
# list_ = dataset.seq_list 


# dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=False, collate_fn=dataset.CNN_collate)

# i = 0
# for x, y, z in dataloader:
#     if len(x) != 256:
#         print('wrong size')
#     i += 1

# x0 = dataset.seq_list[0]
# x = next(iter(dataset))
# x1 = len(next(iter(dataloader))[0])
# y1 = next(iter(dataloader))[1]
# y1 = y1.numpy()
# x1 = torch.transpose(input=x1, dim0=2, dim1=1)
# x1 = x1.numpy()

# total = 256 * 2931 * 852









