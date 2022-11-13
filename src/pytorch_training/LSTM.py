#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 22:52:34 2022

@author: benedict

LSTM script
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# =============================================================================
# Create data
# =============================================================================

N = 100
L = 1000
T = 20

x = np.empty((N, L), np.float32)
x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
y = np.sin(x/1.0/T).astype(np.float32)

plt.figure(figsize=(10, 8))
plt.title('Sin wave')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(x.shape[1]), y[0,:], 'r', linewidth=2.0)
# plt.show()



class LSTMPredictor(nn.Module):
    
    def __init__(self, n_hidden=51):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        # lstm1, lstm2, linear
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)
        
        
    def forward(self, x, future=0):
        # N, 100
        outputs = []
        n_samples = x.size(0)
        
        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        
        
        for input_t in x.split(1, dim=1): # split the tensor into chunks (of 1 in this example)
            # N, 1
            h_t, c_t = self.lstm1(input_t, (h_t, c_t)) # call the first lstm cell with the input
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # call the second cell with the output from the first
            output = self.linear(h_t2) # call the fully connected linear layer on the hidden state from cell 2
            outputs.append(output) # add ouput to list
        
        for i in range(future): # 
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
            
            
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    
    
    
    
if __name__ == "__main__":
    # y = 100, 10000
    
    train_input = torch.from_numpy(y[3:, :-1]) # 97, 999, sequence up to second last value
    train_target = torch.from_numpy(y[3:, 1:]) # 97, 999, sequence from second to the last, shifted by one - we are predicting the next item in the sequence, like sunspot
    
    
    test_input = torch.from_numpy(y[:3, :-1]) # 3, 999
    test_target = torch.from_numpy(y[:3, 1:]) # 3, 999
    
    model = LSTMPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8) # limited memory BFGS, this optimizer can work on all the data - it needs a function as an input
    
    n_steps = 10
    for i in range(n_steps):
        print('Step', i)
        def closure(): # closure for the optimizer
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_target)
            print('loss', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        
        with torch.no_grad():
            future=3000
            pred = model(test_input, future) # calling forward() on our model, with future
            loss = criterion(pred[:, :-future], test_target) # pred also includes the future values now so we need to exclude them here
            print('test loss', loss.item())
            y = pred.detach().numpy()
            
            
            
            
        
        plt.figure(figsize=(12, 6))
        plt.title(f'Step {i + 1}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1] #999
        def draw(y_i, colour):
            plt.plot(np.arange(n), y_i[:n], colour, linewidth=2.0)
            plt.plot(np.arange(n, n+future), y_i[n:], colour + ":", linewidth=2.0)
        # plt.show()        
            
        draw(y[0], 'r')
        draw(y[1], 'b')
        draw(y[2], 'g')
        
        plt.savefig("vis/predict%d.pdf"%i)
        plt.close()
    
            
            
            
            
            