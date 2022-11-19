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
import pickle
import load_data as ld

# =============================================================================
# Fully connected gated recurrent unit neural network with 2 hiden layers
# =============================================================================
class GRU_RNN(nn.Module):
    # =============================================================================
    # class attribtues
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
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
        self.softmax = nn.LogSoftmax(dim=1)
        
       
    # =============================================================================
    # forward propagate input through the constructed network
    # =============================================================================
    def forward(self, x_input):
        # unbatched input dim = (seq_length, input_size)
        h0 = torch.zeros(self.num_layers * 2, self.hidden_size).to(self.device) # init hidden state, as it can't exist before the first forward prop
        out, _ = self.gru(x_input, h0) # output dim = (seq_length, 2 * hidden_size)
        # print(out)
        out = self.fc(out) # output size from the gru network is seq_length, hidden_size, we need to flatten it for the linear layer to 1, hidden_size
        out = self.softmax(out)
        # print(out)
        return out

    

# =============================================================================
# wrapper class for an instance of the GRU_RNN model
# =============================================================================
class GRU_wrapper():
    
    # =============================================================================
    # Class attributes
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = []
    # Hyperparameters
    input_size = 5
    n_classes = 6
    num_layers = 2
    hidden_size = 128
    eta = 0.01
    
    @classmethod
    # =============================================================================
    # Instantiate model, set criterion and optimiser
    # =============================================================================
    def __init__(self, GRU_RNN):
        self.model = GRU_RNN(self.input_size, self.hidden_size, self.num_layers, self.n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        
        # load data using class load_data.py, set to shuffled for now
        self.data_loader = ld.data_loader()
        self.train_data, self.valid_data, self.test_data = self.data_loader.load_shuffled()
        
    @classmethod 
    def class_from_output(self, output, is_tensor):
        # print(output)
        if is_tensor:
            class_index = torch.argmax(output).item()
        else:
            class_index = int(output)
        return self.data_loader.return_classes()[class_index]

    @classmethod
    # =============================================================================
    # training method for one sequence
    # =============================================================================
    def train(self, input_tensor, target_tensor): # training loop that iterates through a sequence, updating parameters after each item in the sequence
        self.model.train()
        # forward propagataion
        # for i in range(input_tensor.size()[0]):
        #     output = self.model(input_tensor[i].to(self.device)) # calling forward() method
            
        for i, vector in enumerate(input_tensor):
            output = self.model(vector.to(self.device)) # calling forward() method
        # back propagation
        self.optimizer.zero_grad() # stop the losses from accumulating training examples
        loss = self.criterion(output, target_tensor) # calcaute loss using the error function (criterion)
        loss.backward() # calculate the gradients aka the loss with respect to the parameters in the computational tree
        self.optimizer.step() # update the parameters based on the grad attribute calculated in the previous line
        return output, loss.item() # returning the last loss and predicted output of the sequence

    @classmethod
    # =============================================================================
    # validation method for one sequence
    # =============================================================================
    def validate(self, input_tensor, target_tensor):
        self.model.eval()
        # forward propagataion
        with torch.no_grad():
            for i, vector in enumerate(input_tensor):
                output = self.model(vector.to(self.device))
            
            # calculate loss and valid_loss
            v_loss = self.criterion(output, target_tensor)
            return output, v_loss.item()
        
    @classmethod
    # =============================================================================
    # Predictions
    # =============================================================================
    def predict(self):
        # helper variables
        # total_correct = 0
        
        print_steps = 200
        plot_steps = 200
        
        t_accuracies = []
        curr_t_loss = 0
        test_loss = []
        t_counter = 0
        t_correct_guesses = 0

        print('Beginning testing')
        # validation loop through validation sequences
        for i, seq in enumerate(self.test_data):
                    
            input_tensor, target_tensor = self.data_loader.seq_to_tensor(seq) 
            target_tensor = target_tensor.type(torch.LongTensor) 
            t_output, t_loss = self.validate(input_tensor.to(self.device), target_tensor.to(self.device))
            t_counter += 1
            guess = self.class_from_output(t_output, True)
            actual = self.class_from_output(target_tensor, False)
            correct = ''
            if guess == actual:
                t_correct_guesses += 1
                correct = 'CORRECT'
            else:
                correct = 'INCORRECT'
                        
            if (i + 1) % (print_steps) == 0:  
                print(f'Test seq number: {i}, test loss = {t_loss}, predicted vs actual: {correct} -> {guess} vs {actual}')

            # for graphing
            curr_t_loss += t_loss
            if (i + 1) % plot_steps == 0:
                test_loss.append(curr_t_loss / plot_steps) # taking averages for graphing
                curr_t_loss = 0
                
        if t_counter != 0:
            print(f'Accuracy of testing epoch: {(t_correct_guesses / t_counter) * 100}')
    
        return t_accuracies, test_loss
    
    @classmethod
    # =============================================================================
    # Train model
    # =============================================================================
    def fit(self, epochs, loss_limit, validate, verbose=True):
        self.model.eval()
        # helper variables
        # total_correct = 0
        print_steps = 200
        plot_steps = 200
        if verbose:
            plot_steps = plot_steps / 20
        accuracies = []
        v_accuracies = []
        curr_t_loss = 0
        curr_v_loss = 0
        train_loss = []
        val_loss = []
 
        
        # per epoch, there will be a training loop through all the trianing data followed by a validation loop through all the validation data
        for epoch in range(epochs):
            counter = 0
            v_counter = 0
            correct_guesses = 0
            v_correct_guesses = 0
            
            # shuffe data each epoch
            self.data_loader.shuffle_data(self.train_data)
            
            # train loop through training sequences
            for i, seq in enumerate(self.train_data):
            
                input_tensor, target_tensor = self.data_loader.seq_to_tensor(seq) # load sequnce into a tensor using custom class load_data()
                target_tensor = target_tensor.type(torch.LongTensor) # convert target tensor to LongTensor for compatibility
                output, loss = self.train(input_tensor.to(self.device), target_tensor.to(self.device)) # call train method for one sequence
                
                # if (loss <= loss_limit): # break the loop if the loss is low enough
                #     print(f'Loss is lower than set limit-> final loss = {loss}')
                #     break
                
                counter += 1
                guess = self.class_from_output(output, True)
                actual = self.class_from_output(target_tensor, False)
                correct = ''
                if guess == actual:
                    correct_guesses += 1
                    correct = 'CORRECT'
                else:
                    correct = 'INCORRECT'

                            
                if (i + 1) % (print_steps) == 0:  
                    print(f'Epoch {epoch}, training seq number: {i}, training loss = {loss}, predicted vs actual: {correct} -> {guess} vs {actual}')
                        
                # for graphing
                curr_t_loss += loss
                if (i + 1) % plot_steps == 0:
                    train_loss.append(curr_t_loss / plot_steps) # taking averages for graphing
                    curr_t_loss = 0
    
            if validate:
                print('Beginning validation')
                # validation loop through validation sequences
                for i, seq in enumerate(self.valid_data):
                    
                    input_tensor, target_tensor = self.data_loader.seq_to_tensor(seq) 
                    target_tensor = target_tensor.type(torch.LongTensor) 
                    v_output, v_loss = self.validate(input_tensor.to(self.device), target_tensor.to(self.device))
                    
                    v_counter += 1
                    guess = self.class_from_output(v_output, True)
                    actual = self.class_from_output(target_tensor, False)
                    correct = ''
                    if guess == actual:
                        v_correct_guesses += 1
                        correct = 'CORRECT'
                    else:
                        correct = 'INCORRECT'
    
                                
                    if (i + 1) % (print_steps) == 0:  
                        print(f'Epoch {epoch}, val seq number: {i}, val loss = {v_loss}, predicted vs actual: {correct} -> {guess} vs {actual}')

                    # for graphing
                    curr_v_loss += v_loss
                    if (i + 1) % plot_steps == 0:
                        val_loss.append(curr_v_loss / plot_steps) # taking averages for graphing
                        curr_v_loss = 0
                        
            if counter != 0:
                print(f'Accuracy of training epoch: {(correct_guesses / counter) * 100}')
                print(f'Accuracy of validation set: {(v_correct_guesses / v_counter) * 100}')
                accuracies.append((correct_guesses / counter) * 100)
                v_accuracies.append((v_correct_guesses / v_counter) * 100)
        self.history.append([accuracies, v_accuracies, train_loss, val_loss])
                    
    @classmethod
    # =============================================================================
    # method to save model to state_dict
    # =============================================================================
    def save_model(self, model_name):
        torch.save(self.model.state_dict(), f'saved_models/{model_name}.pt')
        print(f'{model_name} state_dict successfully saved')

    @classmethod
    # =============================================================================
    # method to load model from state_dict
    # =============================================================================
    def load_model(self, model_name):
        self.model.load_state_dict(torch.load(f'saved_models/{model_name}.pt'))
        print(f'{model_name} state dictionary successfully loaded')
        print(self.model.eval())

    @classmethod
    # =============================================================================
    # method to print params of model
    # =============================================================================
    def print_params(self):
        params = self.model.parameters()
        for p in params:
            print(p)
            
    @classmethod
    # =============================================================================
    # returns history
    # =============================================================================            
    def return_history(self):
        return self.history
            
    @classmethod
    # =============================================================================
    # method that saves history to a pkl file
    # =============================================================================            
    def save_history(self, version_number):
        with open(f'pkl/GRU_RNN_v{version_number}_history.pkl', 'wb') as f:
            pickle.dump(self.history, f)
            print('History saved')


# =============================================================================
# instantiate model and wrapper then train and save
# =============================================================================
model = GRU_wrapper(GRU_RNN)
model.fit(epochs=10, loss_limit=0, validate=True, verbose=False)
model.save_model('GRU_RNN_v4')



"""
Paremeters for GRU_RNN_v2:
tensor([-0.2460,  0.1251,  0.0764,  0.0246, -0.3283,  0.1671],
       requires_grad=True)
"""



# plt.figure()
# plt.plot(train_losses)
# plt.show()


# =============================================================================
# instantiate model and wrapper then load
# =============================================================================
# model = GRU_wrapper(GRU_RNN)
# model.load_model('GRU_RNN_v3')
# model.predict()

"""
DOCUMENTATION: 
TEST 1:
    Model: GRU-RNN_v2 -> Hyperparameters:
        2 layered GRU with 128 hidden units
        1 layer fully connected with 128 units
        Learning rate = 0.001
        Optimiser = Adam
        Loss = CrossEntropyLoss
    Data: Evenly distributed, uninterpolated, featurised, threshold for sequence length of 4 (must be greater)
    RESULTS:
        Test accuracy on randomseed=15 is 66.39%      
        Common sense baseline for evenly distributed classes is 16.67%.

TEST 2:
    Model: GRU-RNN_v3 -> Hyperparameters:
        2 layered GRU with 128 hidden units
        1 layer fully connected with 128 units
        Learning rate = 0.0003 -> CHANGED
        Optimiser = Adam
        Loss = CrossEntropyLoss    
    Data: Evenly distributed, uninterpolated, featurised, threshold for sequence length of 1 (must be greater), i.e. no threshold -> CHANGED
    RESULTS: 
        Test accuracy on randomseed=15 is 59.02%
 
TEST 3:
    Model: GRU-RNN_v4 -> Hyperparameters:
        2 layered GRU with 128 hidden units
        1 layer fully connected with 128 units
        Learning rate = 0.01 -> CHANGED
        Optimiser = Adam
        Loss = CrossEntropyLoss    
    Data: Evenly distributed, uninterpolated, featurised, threshold for sequence length of 1 (must be greater), i.e. no threshold -> CHANGED
    RESULTS: 
        Test accuracy on randomseed=15 
        
  
    
Introduce all data!

"""
























"""
Common sense baseline: Based on random decision, what is the outcome? Compare to this common sense baseline / statistical power
Get a working prototype, a small model with statistical power
Regularisation to deal with overfitting

More data is better, despite imbalance - statisitcal power is harder to achieve with imbalanced

Common sense baseline:
    Common sense would be the probability of predicting one class all the time
    Find methods for dealing with imalance
    
    Maybe don't remove straight line courses
    Prove my a priori knowledge is correct!
    
    
    
    
    
Mark scheme:
    Functionality of the project is to a working level
    Functioning prototype
    
    
GAN
Time series data augmentation




"""







        
