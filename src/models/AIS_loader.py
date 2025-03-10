#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:18:55 2022
@author: Benedict Grey

Script implementing customised PyTorch Dataset class

Data fields:
<--------------------------------------------------------------------->
    Varying time series:
        speed | lat | lon | delta_time | delta_course | target
        
    Linearly interpolated time series:
        speed | course | lat | lon | desired
        
    Varying for non-linear attention interpolation:
        speed | lat | lon | delta_time_cum | delta_course | target
<--------------------------------------------------------------------->
"""
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

class AIS_loader(Dataset):
    
# =============================================================================
#     structure of input: list of sequences (shuffled already)
#     each sequence is: sequence length x features
#     feature columns: speed, lat, lon, time_delta, course_delta
#     input shape: batch_size, n_features, seq_length
# =============================================================================
    def __init__(self, choice, split, version, split_2=None):
        self.choice = choice
        random.seed(15) # set random seed to reproduce random results
        with open(f'../../data/pkl/{choice}/{split}_v{version}.pkl', 'rb') as f:
            self.seq_list = pickle.load(f)
        if split_2 != None:
            with open(f'../../data/pkl/{choice}/{split_2}_v{version}.pkl', 'rb') as f:
                self.seq_list += pickle.load(f)
        with open(f'../../data/pkl/{choice}/utils_v{version}.pkl', 'rb') as f:
            obj = pickle.load(f)
            self.n_features = obj[0]
            # self.n_features = 4
            self.n_classes = obj[1]
            self.seq_length = obj[2]
            
    def __getitem__(self, index):
        sequence = self.seq_list[index]
        features = sequence[:, :-1]
        # features = features[:, [0, 3, 4]]
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
    
        sequences = torch.zeros(len(data), int(self.seq_length), int(self.n_features)) # create empty padded tensor to fill in loop below
        labels = torch.tensor(labels)
        lengths = torch.tensor(lengths)
        
        for i in range(len(data)):
            j, k = data[i][0].size(0), data[i][0].size(1)
            sequences[i] = torch.cat([data[i][0], torch.zeros((int(self.seq_length) - j, k))])
        
        sequences = torch.transpose(input=sequences, dim0=1, dim1=2) # transpose sequences
        return sequences.float(), labels.long(), lengths.long()
    
    # =============================================================================
    # same as CNN_collate but doesn't transpose the matrix 
    # =============================================================================
    def GRU_collate(self, data):
        seqs, labels, lengths = zip(*data)
        int(self.seq_length)
    
        sequences = torch.zeros(len(data), int(self.seq_length), self.n_features) # create empty padded tensor to fill in loop below
        labels = torch.tensor(labels)
        lengths = torch.tensor(lengths)
        
        for i in range(len(data)):
            j, k = data[i][0].size(0), data[i][0].size(1)
            sequences[i] = torch.cat([data[i][0], torch.zeros((int(self.seq_length) - j, k))])
        
        if self.choice == 'non_linear':
            sequences = sequences[:, :, [0,1,2,4]]
            time_steps = sequences[:, :, 3]
            return sequences.float(), time_steps.float(), labels.long(), lengths.long()
            
        else:
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
# driver code
# =============================================================================
# =============================================================================
# def main():
#     dataset = AIS_loader(choice='non_linear', split='test', version=1)
#     dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, collate_fn=(dataset.GRU_collate))
#     dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, collate_fn=(dataset.CNN_collate))
#     dataset.visualise_tensor(250)
#     
#     for batch_seqs, batch_time_steps, labels, lengths in dataloader:
#         feature = batch_seqs[0]
#         print(batch_time_steps.shape)
#         time_steps = batch_time_steps[0]
#         # print(feature)
#         print(time_steps)
#         # print(batch_seqs.shape)
#         break
#     
#     for batch_seqs, labels, lengths in dataloader:
#         feature = batch_seqs[0]
#         # print(batch_time_steps.shape)
#         # time_steps = batch_time_steps[0]
#         print(batch_seqs.shape)
#         break
#         
#         
#         # plot for GRUs
#         plt.plot(feature[:, 2], feature[:, 3])
#         plt.scatter(feature[:, 2], feature[:, 3], s=8)
#         plt.show()
#         
#         # plot for CNNs
#         plt.plot(feature[2, :], feature[3, :])
#         plt.scatter(feature[2, :], feature[3, :], s=8)
#         plt.show()
#         break
#         
# =============================================================================
    # print(sample_item)


# if __name__ == "__main__":
    # main()







