#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:54:04 2022
@author: BenedictGrey

PyTorch demo

"""

import torch # PyTorch to create tensors to store all the numerical values for data, weights and bias
import torch.nn as nn # use to make the weight and bias tensors part of the neural network
import torch.nn.functional as F # gives is the activation functions
from torch.optim import SGD # stochastic gradient descent for optimisation

import matplotlib.pyplot as plt
import seaborn as sns # west wing sam seaborn

class BasicNN(nn.Module): # new class inheriting from pytorch class Module
    
    def __init__(self): # contructor or initaliser method to create and initiliases the weights and biases
        super().__init__() # parent initialiser to automate 
        
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False) # add weight as parameter to the network
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False) 
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False) # add weight as parameter to the network
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False) 
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False) 
        
        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)
        
        
    def forward(self, input): # forward propagation
    
        # Input to hidden connections (weights * inputs + bias)
        input_to_top_relu = input * self.w00 + self.b00
        input_to_bottom_relu = input * self.w10 + self.b10
        
        # Activations happening in hidden nodes
        top_relu_output = F.relu(input_to_top_relu)
        bottom_relu_output = F.relu(input_to_bottom_relu)
        
        # Hidden to output connections (hidden_outputs * weights)
        scaled_top_relu_output = top_relu_output * self.w01
        scaled_bottom_relu_output = bottom_relu_output * self.w11
        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
        
        # output from the final node
        output = F.relu(input_to_final_relu)
        
        return output
        

input_doses = torch.linspace(start=0, end=1, steps=11)
model = BasicNN()    

output_values = model(input_doses)

print(output_values)

sns.set(style='whitegrid')

sns.lineplot(x=input_doses,
             y = output_values,
             color='green',
             linewidth=2.5)


