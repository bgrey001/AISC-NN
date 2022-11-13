#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:57:44 2022

@author: benedict

Script for building and training a GRU-RNN
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import load_data as ld

# =============================================================================
# Fully connected gated recurrent unit neural network with 2 hiden layers
# =============================================================================
class GRU_RNN(nn.Module):
    # =============================================================================
    # class attribtues
    # =============================================================================
    device = torch.device('cuda')
    
    
    # =============================================================================
    # constructor
    # =============================================================================
    def __init__(self, input_size, hidden_size, num_layers, n_classes):
        
        super(GRU_RNN, self).__init__()
        self.hidden_size = hidden_size # number of nodes in the hidden state
        self.num_layers = num_layers # number of recurrent layers, more than one means stacking GRUs
        
        # build network
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True) # the GRU-RNN, using bidirection according to Weerakody's research 
        self.fc = nn.Linear(hidden_size * 2, n_classes) # prediction layer for classification
        self.softmax = nn.Softmax()
        
       
    # =============================================================================
    # forward propagate input through the constructed network
    # =============================================================================
    def forward(self, x_input):
        # unbatched input dim = (seq_length, input_size)
        h0 = torch.zeros(self.num_layers * 2, self.hidden_size).to(self.device) # init hidden state, as it can't exist before the first forward prop
        out, _ = self.gru(x_input, h0) # output dim = (seq_length, 2 * hidden_size)
        out = self.fc(self.softmax(out)) # output size from the gru network is seq_length, hidden_size, we need to flatten it for the linear layer to 1, hidden_size
        return out


    
    
    

# =============================================================================
# wrapper class for an instance of the GRU_RNN model
# =============================================================================
class GRU_wrapper():
    
    # =============================================================================
    # Hyperparameter attributes
    # =============================================================================
    input_size = 5
    n_classes = 6
    num_layers = 2
    hidden_size = 64
    eta = 0.001
    
    @classmethod
    # =============================================================================
    # Instantiate model, set criterion and optimiser
    # =============================================================================
    def __init__(self, GRU_RNN):
        self.device = torch.device('cuda')
        self.model = GRU_RNN(self.input_size, self.hidden_size, self.num_layers, self.n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        
        # load data using class load_data.py, set to shuffled for now
        self.data_loader = ld.data_loader()
        self.train_data, self.valid_data, self.test_data = self.data_loader.load_shuffled()

    @classmethod
    # =============================================================================
    # function for training the model on a single sequence
    # =============================================================================
    def train(self, input_tensor, target_tensor): # training loop that iterates through a sequence, updating parameters after each item in the sequence

        self.model.train()
        # forward propagataion
        for i in range(input_tensor.size()[0]):
            output = self.model(input_tensor[i].to(self.device))
        # back prop
        self.optimizer.zero_grad() # stop the losses from accumulating training examples
        self.loss = self.criterion(output, target_tensor) # calcaute loss using the error function (criterion)
        self.loss.backward() # calculate the gradients aka the loss with respect to the parameters in the computational tree
        self.optimizer.step() # update the parameters based on the grad attribute calculated in the previous line
        return output, self.loss.item()
    
    @classmethod
    # =============================================================================
    # Train model
    # =============================================================================
    def fit(self):
        self.model.eval()
        # helper variables
        epochs = 10
        curr_loss = 0
        # total_correct = 0
        print_steps = 10
        plot_steps = 100
        list_loss = []
        
        # loop through epochs and then through sequences
        for i in range(epochs):
            for j, seq in enumerate(self.train_data):
                input_tensor, target_tensor = self.data_loader.seq_to_tensor(seq)
                target_tensor = target_tensor.type(torch.LongTensor)
                
                output, loss = self.train(input_tensor.to(self.device), target_tensor.to(self.device))
                curr_loss += loss
                
                if (j + 1) % print_steps == 0:    
                    print(f'Epoch: {i}, seq number: {j}, loss = {loss}')
                    
                if (i + 1) % plot_steps == 0:
                    list_loss.append(curr_loss / plot_steps) # taking averages
                    curr_loss = 0
        
    @classmethod
    # =============================================================================
    # function to save model
    # =============================================================================
    def save_model(self, model_name):
        torch.save(self.model.state_dict(), f'saved_models/{model_name}.pt')
        print(f'{model_name} state dictionary successfully saved')


d



# =============================================================================
# instantiate classes
# =============================================================================
model = GRU_wrapper(GRU_RNN)
# model.fit()
# model.save_model('test_model')











        
