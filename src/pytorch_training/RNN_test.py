#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:37:35 2022

@author: benedict
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


from utils import ALL_LETTERS, N_LETTERS # attributes, or variables
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example, generate_test_set # methods or functions


# =============================================================================
# initialise the model to test here
# =============================================================================

class RNN(nn.Module): # inherit from nn.Module
    
    def __init__(self, input_size, hidden_size, output_size): # creates the structure of the network with no real values in it
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        # nn.Linear creates a fully connected layey with no activation function, the activation function must be applied to this layer seperately in the forward method
        #self.i2x = nn.Linear(in_features=, out_features=)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # combined to hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # combined to output
        self.softmax = nn.LogSoftmax(dim=1) # 1, 57 -> we need the second dimension i.e. 57
        
        
    def forward(self, input_tensor, hidden_tensor): # this is different to a feed forward NN where there is only one input, now we input the hidden state as well
        # combine the input and hidden tensors, process them through the linear layers and then softmax one for output and return both output and hidden
        
        combined = torch.cat((input_tensor, hidden_tensor), 1).to('cuda')
        
        hidden = self.i2h(combined) # process the input and hidden tensors through the fully connected layer
        output = self.i2o(combined) # process hidden output through the fc layer
        output = self.softmax(output) # apply activation function
    
        return output, hidden
    
        
    def init_hidden(self): # need initial hidden state in the beginning
        return torch.zeros(1, self.hidden_size)
    
    
    

# =============================================================================
# load the necessary data
# =============================================================================
category_lines, all_categories = load_data()
n_categories = len(all_categories) # number of classes for this classification task which is 18
n_hidden = 128 # hyperparameter to be tuned

# =============================================================================
# instantiate the model
# =============================================================================
loaded_model = RNN(N_LETTERS, n_hidden, n_categories).to('cuda') # running the __init__() method to create the structure of the network, nothing has happened yet
loaded_model.eval()
# for param in loaded_model.parameters():
#     print(param)

# =============================================================================
# load the model state dictionary to give it the trained parameters
# =============================================================================
loaded_model.load_state_dict(torch.load('models/rnn.pt'))
# for param in loaded_model.parameters():
#     print(param)


def category_from_output(output): # likelihood of each category, so we are going to return the index of the highest probablity
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]



# load test set

def test_model(set_size):
    test_set = generate_test_set(category_lines, all_categories, set_size)
    n_correct = 0
    for x in test_set:
        target = x[0]
        #x_seq = x[1]
        #category_tensor = x[2]
        line_tensor = x[3]
        hidden = loaded_model.init_hidden()
    
    # =============================================================================
    #     print(category_tensor.numpy())
    #     print(target)
    # =============================================================================
        
        for i in range(line_tensor.size()[0]): # process a single sequence in this for loop
            output, hidden = loaded_model(line_tensor[i].to('cuda'), hidden.to('cuda')) # hidden state is being updated each iteration as is the output // forward passes!
    
        guess = category_from_output(output)
        if guess == target:
            n_correct += 1
    # =============================================================================
    #     print(f'Predicted: {guess} \nTarget: {target}')
    # =============================================================================
        
# =============================================================================
#     print(f'Accuracy = {n_correct/set_size * 100}')
# =============================================================================
    return n_correct/set_size * 100






















