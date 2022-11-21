#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 09:24:55 2022

@author: benedict

1DCNN model

Time sequence length is the height of the input matrix

Convolution kernels always have the same width as the time series (number of features), while their length can be varied (lenght of time sequence)

The kernel moves in one direction from the beginning of the time series to the end, unlike the right left and up down movement of a 2DCNN





"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import pickle
import load_data as ld



class CNN_1D(nn.Module):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    def __init__(self, n_features, hidden_size, n_classes):
        super(CNN_1D, self).__init__()
        
        # self.batch_norm1 = nn.BatchNorm1d(self.n_features)
        # self.dropout1 = nn.Dropout(0.1)
        
        # conv layers 1
        self.conv_1 = nn.utils.weight_norm(nn.Conv1d(in_channels=n_features, out_channels=hidden_size, kernel_size=(3)))
        self.relu_1 = nn.ReLU()
        self.pool_1= nn.MaxPool1d(2)
        
        # conv layers 2
        self.batch_norm_2 = nn.BatchNorm1d(16)
        self.dropout_2 = nn.Dropout(0.1)
        self.conv_2 = nn.utils.weight_norm(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(3)))
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool1d(2)
        
        # conv layers 3
        self.batch_norm_3 = nn.BatchNorm1d(32)
        self.dropout_3 = nn.Dropout(0.1)
        self.conv_3 = nn.utils.weight_norm(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=(3)))
        self.relu_3 = nn.ReLU()
        self.pool_3 = nn.MaxPool1d(2)
        
        # flatten and prediction layers
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(1024, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        
        
        
    def forward(self, input_x):
        
        # conv layers 1
        input_x = self.conv_1(input_x)
        input_x = self.relu_1(input_x)
        input_x = self.pool_1(input_x)
        
        # conv layers 2
        input_x = self.batch_norm_2(input_x)
        input_x = self.dropout_2(input_x)
        input_x = self.conv_2(input_x)
        input_x = self.relu_2(input_x)
        input_x = self.pool_2(input_x)
        
        # conv layers 3
        input_x = self.batch_norm_3(input_x)
        input_x = self.dropout_3(input_x)
        input_x = self.conv_3(input_x)
        input_x = self.relu_3(input_x)
        input_x = self.pool_3(input_x)
        
        # flatten and prediction layers
        input_x = self.flatten(input_x)
        input_x = F.relu(self.fc_1(input_x))
        input_x = F.relu(self.fc_2(input_x))
        input_x = self.fc_3(input_x)
        output = self.softmax(input_x)
        return output
        
    
# =============================================================================
# wrapper class for an instance of the GRU_RNN model
# =============================================================================    
class CNN_1D_wrapper():
    
    # =============================================================================
    # Class attributes
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    
    history = []
    training_losses = []
    validation_losses = []
    test_losses = []
    
    training_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    
    # =============================================================================
    # Hyperparameters
    # =============================================================================
    n_features = 4
    hidden_size = 16
    n_classes = 6
    eta = 3e-5
    alpha = 1e-4
    weight_decay = 1e-5

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    def __init__(self, CNN_1D, optimizer):
        
        self.model = CNN_1D(n_features=self.n_features, hidden_size=self.hidden_size, n_classes=self.n_classes).to(self.device)
        
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.eta, weight_decay=self.weight_decay, momentum=self.alpha)
        elif optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.eta, weight_decay=self.weight_decay, momentum=self.alpha)

        self.criterion = nn.CrossEntropyLoss()
        
        self.data_loader = ld.data_loader(choice='linear_interp', version='2')
        self.train_data, self.valid_data, self.test_data = self.data_loader.load_shuffled()
        
    def class_from_output(self, output, is_tensor):
        # print(output)
        if is_tensor:
            # print(output)
            class_index = torch.argmax(output).item()
        else:
            class_index = int(output)
        return self.data_loader.return_classes()[class_index]
    
    
    # =============================================================================
    # Train model
    # =============================================================================
    def fit(self, epochs, validate):
        # helper variables
        train_print_steps = 800
        val_print_steps = 200
        # plot_steps = 200
        
        train_loss = 0
        valid_loss = 0
        
        # load data
        train_batches = self.data_loader.load_batch_shuffled(data_list=self.train_data, batch_size=self.batch_size)
        valid_batches = self.data_loader.load_batch_shuffled(data_list=self.valid_data, batch_size=self.batch_size)
 
        # per epoch, there will be a training loop through all the trianing data followed by a validation loop through all the validation data
        for epoch in range(epochs):
            
            correct = 0
            v_correct = 0
            train_loss_counter = 0
            valid_loss_counter = 0
            
            # shuffe data each epoch
            self.data_loader.shuffle_data(self.train_data)
            self.data_loader.shuffle_data(self.valid_data) 

            
            # =============================================================================
            # training loop
            # =============================================================================
            for i, batch in enumerate(train_batches):
                self.model.train()
                
                # forward propagation
                input_tensor, target_tensor = self.data_loader.batch_cnn_seq(batch)
                input_tensor = input_tensor.float()
                output = self.model(input_tensor.to(self.device))
                target_tensor = target_tensor.type(torch.LongTensor).to(self.device)  # convert target tensor to LongTensor for compatibility

                # backpropagation
                self.optimizer.zero_grad()
                loss = self.criterion(output, target_tensor)
                loss.backward()
                self.optimizer.step()
                
                outputs = torch.argmax(output, dim=1) 
                correct += (outputs == target_tensor).sum().item()
                            
                if (i + 1) % (train_print_steps) == 0:  
                    print(f'Epoch {epoch}, batch number: {i + 1}, training loss = {loss}')
                    train_loss_counter += 1
                    train_loss += loss.item()
                    
            train_accuracy = 100 * (correct / (len(train_batches) * self.batch_size))
            self.training_accuracies.append(train_accuracy)
            self.training_losses.append(train_loss/train_loss_counter)        
            print("========================================================================================================================================================== \n" +
                  f"------------------> training accuracy = {train_accuracy}, average training loss = {train_loss / train_loss_counter} <------------------\n" +
                  "========================================================================================================================================================== \n")     
                
       
            train_loss = 0
            train_loss_counter = 0
            
            
            # =============================================================================
            # validation loop
            # =============================================================================
            if validate:
                for i, batch in enumerate(valid_batches):
                    self.model.eval()
                    with torch.no_grad():
                        # forward propagataion
                        input_tensor, target_tensor = self.data_loader.batch_cnn_seq(batch)
                        input_tensor = input_tensor.float()
                        v_output = self.model(input_tensor.to(self.device))
                        target_tensor = target_tensor.type(torch.LongTensor).to(self.device)  # convert target tensor to LongTensor for compatibility
                        
                        # calculate loss and valid_loss
                        v_loss = self.criterion(v_output, target_tensor)
                        v_outputs = torch.argmax(v_output, dim=1) 
                        v_correct += (v_outputs == target_tensor).sum().item()
                                    
                        if (i + 1) % (val_print_steps) == 0:  
                            print(f'Epoch {epoch}, validation batch number: {i + 1}, validation loss = {v_loss}')
                            valid_loss_counter += 1
                            valid_loss += v_loss.item()
                
                val_accuracy = 100 * (v_correct / (len(valid_batches) * self.batch_size))
                self.validation_accuracies.append(val_accuracy)
                self.validation_losses.append(valid_loss/valid_loss_counter) 
                print("========================================================================================================================================================== \n" +
                      f" ------------------> validation accuracy = {val_accuracy}, average validation loss = {valid_loss / valid_loss_counter} <------------------" +
                      "========================================================================================================================================================== \n")     
                    
                valid_loss = 0
                valid_loss_counter = 0
                        
        # history
        self.history.append([self.training_accuracies, self.training_losses, self.validation_accuracies, self.validation_losses])
            
        
        
        
    # =============================================================================
    # method that tests model on test data
    # =============================================================================  
    def predict(self):
        
        test_correct = 0
        test_loss_counter = 0
        test_loss = 0
        test_print_steps = 100
        test_batches = self.data_loader.load_batch_shuffled(data_list=self.valid_data, batch_size=self.batch_size)
        
        for i, batch in enumerate(test_batches):
            self.model.eval()
            with torch.no_grad():
                # forward propagataion
                input_tensor, target_tensor = self.data_loader.batch_cnn_seq(batch)
                input_tensor = input_tensor.float()
                test_output = self.model(input_tensor.to(self.device))
                target_tensor = target_tensor.type(torch.LongTensor).to(self.device)  # convert target tensor to LongTensor for compatibility
                
                # calculate loss and valid_loss
                t_loss = self.criterion(test_output, target_tensor)
                test_outputs = torch.argmax(test_output, dim=1) 
                test_correct += (test_outputs == target_tensor).sum().item()
                            
                if (i + 1) % (test_print_steps) == 0:  
                    print(f'test batch number: {i + 1}, test loss = {t_loss}')
                    test_loss_counter += 1
                    test_loss += t_loss.item()
                    
        test_accuracy = 100 * (test_correct / (len(test_batches) * self.batch_size))
        self.test_losses.append(test_loss/test_loss_counter) 
        self.test_accuracies.append(test_accuracy)
        print("========================================================================================================================================================== \n" +
              f" ------------------> test accuracy = {test_accuracy}, average test loss = {test_loss / test_loss_counter} <------------------" +
              "========================================================================================================================================================== \n")     
            
        test_loss = 0
        test_loss_counter = 0

                     
        # history
        self.history.append([self.test_accuracies, self.test_losses])
            
        
        
        
    # =============================================================================
    # method to save model to state_dict
    # =============================================================================
    def save_model(self, version_number):
        torch.save(self.model.state_dict(), f'saved_models/CNN_1D_v{version_number}.pt')
        print(f'CNN_1D_v{version_number} state_dict successfully saved')

    # =============================================================================
    # method to load model from state_dict
    # =============================================================================
    def load_model(self, model_name):
        self.model.load_state_dict(torch.load(f'saved_models/{model_name}.pt'))
        print(f'{model_name} state dictionary successfully loaded')
        print(self.model.eval())

    # @classmethod
    # =============================================================================
    # method to print params of model
    # =============================================================================
    def print_params(self):
        params = self.model.parameters()
        for p in params:
            print(p)
            
    # @classmethod
    # =============================================================================
    # returns history
    # =============================================================================            
    def return_history(self):
        return self.history
            
    # @classmethod
    # =============================================================================
    # method that saves history to a pkl file
    # =============================================================================            
    def save_history(self, version_number):
        with open(f'pkl/CNN_1D_v{version_number}_history.pkl', 'wb') as f:
            pickle.dump(self.history, f)
            print('History saved')
      
    # =============================================================================
    # plot given metric
    # =============================================================================
    def plot(self, metric):
        plt.figure()
        plt.plot(metric)
        plt.show()


            
            
# =============================================================================
# instantiate model and wrapper then train and save
# =============================================================================
model = CNN_1D_wrapper(CNN_1D, optimizer='Adam')
model.fit(epochs=25, validate=True)
model.save_model('1')
model.save_history('1')
# history = model.return_history()
# model.plot(history[0])
    
# =============================================================================
# instantiate model and wrapper then load
# =============================================================================
# model = CNN_1D_wrapper(CNN_1D, optimizer='Adam')
# model.load_model('CNN_1D_v1')
# model.predict()
# history = model.return_history()
# model.plot(history[0][1])


    
"""
DOCUMENTATION: 
TEST 1:
    Model: CNN_1D_v1 -> Hyperparameters:
        
        
        Learning rate = 1e-2 or 0.01
        Optimiser = Adam
        Loss = CrossEntropyLoss
        Batch size = 32
        Epochs = 25
    Data: Linearly interpolated, intervals of 10 minutes
    RESULTS:
        Test train_accuracy on randomseed=15 is 81.98%      
        Common sense baseline for unevenly distributed classes is 57.77%.
"""
    
    
    
    