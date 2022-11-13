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



with open('../../data/pkl/train.pkl', 'rb') as f:
    train_seq_list = pickle.load(f)
    
with open('../../data/pkl/valid.pkl', 'rb') as f:
    valid_seq_list = pickle.load(f)

with open('../../data/pkl/test.pkl', 'rb') as f:
    test_seq_list = pickle.load(f)
    
    
n_features = 5

seq = train_seq_list[0]


    
# =============================================================================
# takes a sequence from a sequence_list and returns an input_tensor and target tensor ready for the network
# =============================================================================
def seq_to_tensor(seq): 
    input_tensor = torch.zeros(len(seq), 1, n_features)
    target_tensor = torch.zeros(len(seq), 1)
    for i, item in enumerate(seq):
        input_tensor[i][0][:] = torch.tensor(item[:-1])
        target_tensor[i][0] = input_tensor[i][0][-1]
    return input_tensor, target_tensor






# print(input_tensor)


# def seq_to_tensor(item): 
#     tensor = torch.zeros(len(item), 1, num_features)
#     for i, letter in enumerate(item): # the enumerate() function adds a counter as the key of tghe enumerate object (for iterating through tuples when access to an index is needed)
#         tensor[i][0][:] = torch.tensor(item[0, :])
#     return tensor

# input_tensor, target_tensor = seq_to_tensor(seq)


# =============================================================================
# 
# def target_to_tensor(seq):
#     tensor = torch.zeros(len(seq), n_features)
#     
#     
# def generate_input_target_tensors(seq_list):
# =============================================================================
    
# =============================================================================
# now we train
# =============================================================================
    
# current_loss = 0
# all_losses = []

# total_correct = 0

# plot_steps, print_steps = 1000, 1000
# n_epochs = 100

# for i in range(n_epochs):
#     # category, line, category_tensor, line_tensor = random_training_example(category_lines=category_lines, all_categories=all_categories)
    
#     # line_tensor is one word or sequence, category tensor is one sequence lengths of targets
#     for seq in train_seq_list:
#         input_tensor, target_tensor = seq_to_tensor(seq)
        
#         # need to get line tensor and category tensor
        
#         output, loss = train(input_tensor.to('cuda'), target_tensor.to('cuda'))
#         current_loss += loss
        
#         if (i + 1) % plot_steps == 0:
#             all_losses.append(current_loss / plot_steps) # taking averages
#             current_loss = 0
            
#         if (i + 1) % print_steps == 0:
#             guess = category_from_output(output)
#             correct = "CORRECT" if guess == category else f"WRONG ({category})"
#         #         print(f'{i} {i/n_iterations*100} {loss:.4f} {line} / {guess} {correct}')
#             print(f'{i} {loss:.4f} {line} / {guess} {correct}')
            
        
    
    
    
    


# =============================================================================
# function for converting nparrays of individual sequences into tensors to serve as input for RNNs
# item = pole_and_line_list_test_seq[1]
# num_features = 6
# 
# 
# def seq_to_tensor(item): 
#     tensor = torch.zeros(len(item), 1, num_features)
#     for i, letter in enumerate(item): # the enumerate() function adds a counter as the key of tghe enumerate object (for iterating through tuples when access to an index is needed)
#         tensor[i][0][:] = torch.tensor(item[0, :])
#     return tensor
# 
# input_tensor = seq_to_tensor(item)
# print(input_tensor)
# =============================================================================



    
    
    
    
    
# =============================================================================
# total number of inputs
# sum = 0
# 
# for item in train_seq_list:
#     sum += len(item)
#     
# print(sum)
# =============================================================================
