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

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import load_data as ld


# =============================================================================
# model class inherits from torch Module class
# =============================================================================
class CNN_1D(nn.Module):

    # =============================================================================
    # class attributes
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================================================================
    # constructor
    # =============================================================================
    def __init__(self, n_features, n_classes, seq_length, conv_L1, kernel_size, pool_size):
        super(CNN_1D, self).__init__()

        # calculate channel sizes for the different convolution layers
        conv_L2 = 2 * conv_L1
        conv_L3 = 2 * conv_L2

        # conv layers 1
        self.batch_norm_1 = nn.BatchNorm1d(n_features)
        self.dropout_1 = nn.Dropout(0.1)
        # self.conv_1 = nn.utils.weight_norm(nn.Conv1d(
        #     in_channels=n_features, out_channels=conv_L1, kernel_size=kernel_size))
        # self.relu_1 = nn.ReLU()
        # self.pool_1 = nn.MaxPool1d(pool_size)

        self.conv_1 = nn.Conv1d(in_channels=n_features, out_channels=conv_L1, kernel_size=kernel_size)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool1d(pool_size)    

        # conv layers 2
        # self.batch_norm_2 = nn.BatchNorm1d(conv_L1)
        # self.dropout_2 = nn.Dropout(0.1)
        # self.conv_2 = nn.utils.weight_norm(
        #     nn.Conv1d(in_channels=conv_L1, out_channels=conv_L2, kernel_size=kernel_size))
        # self.relu_2 = nn.ReLU()
        # self.pool_2 = nn.MaxPool1d(pool_size)
        # self.conv_2 = nn.Conv1d(in_channels=conv_L1, out_channels=conv_L2, kernel_size=kernel_size)
        # self.relu_2 = nn.ReLU()
        # self.pool_2 = nn.MaxPool1d(pool_size)

        # conv layers 3
        # self.batch_norm_3 = nn.BatchNorm1d(conv_L2)
        # self.dropout_3 = nn.Dropout(0.1)
        # self.conv_3 = nn.utils.weight_norm(
            # nn.Conv1d(in_channels=conv_L2, out_channels=conv_L3, kernel_size=kernel_size))
        # self.relu_3 = nn.ReLU()
        # self.pool_3 = nn.MaxPool1d(pool_size)

        # configure transformed dimensions of the input as it reaches the fully connected layer
        # conv_L3_dim = math.floor((math.floor((math.floor((seq_length - (kernel_size - 1)) /
        #                          pool_size) - (kernel_size - 1)) / pool_size) - (kernel_size - 1)) / pool_size)
        # flat_size = conv_L3 * conv_L3_dim
        
        
        conv_L1_dim = math.floor((seq_length - (kernel_size - 1))/ pool_size)
        conv_L2_dim = math.floor((conv_L1_dim - (kernel_size - 1)) / pool_size)
        conv_L3_dim = math.floor((conv_L2_dim - (kernel_size - 1)) / pool_size)
        
        
        flat_size = conv_L1 * conv_L1_dim

        # flatten and prediction layers
        self.flatten = nn.Flatten()
        # self.fc_1 = nn.Linear(flat_size, 128)
        # self.fc_2 = nn.Linear(128, 256)
        # self.fc_3 = nn.Linear(256, 128)
        # self.fc_4 = nn.Linear(128, n_classes)

        self.fc_1 = nn.Linear(flat_size, 64)
        self.fc_2 = nn.Linear(64, 6)
        # self.fc_3 = nn.Linear(64, 6)
        self.softmax = nn.LogSoftmax(dim=1)

    # =============================================================================
    # forward propagation method
    # =============================================================================

    def forward(self, input_x):

        # conv layers 1
        input_x = self.batch_norm_1(input_x)
        input_x = self.dropout_1(input_x)
        input_x = self.conv_1(input_x)
        input_x = self.relu_1(input_x)
        input_x = self.pool_1(input_x)

        # conv layers 2
        # input_x = self.batch_norm_2(input_x)
        # input_x = self.dropout_2(input_x)
        # input_x = self.conv_2(input_x)
        # input_x = self.relu_2(input_x)
        # input_x = self.pool_2(input_x)

        # # conv layers 3
        # input_x = self.batch_norm_3(input_x)
        # input_x = self.dropout_3(input_x)
        # input_x = self.conv_3(input_x)
        # input_x = self.relu_3(input_x)
        # input_x = self.pool_3(input_x)

        # flatten and prediction layers
        input_x = self.flatten(input_x)
        input_x = F.relu(self.fc_1(input_x))
        input_x = F.relu(self.fc_2(input_x))
        # input_x = F.relu(self.fc_3(input_x))
        # input_x = self.fc_3(input_x)
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

    version_number = 0

    history = {'training_accuracy': [], 'training_loss': [], 'validation_accuracy': [], 'validation_loss': [], 'test_accuracy': [], 'test_loss': []}
    training_losses = []
    validation_losses = []
    test_losses = []
    training_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    # =============================================================================
    # Data attributes
    # =============================================================================
    datatype = 'linear_interp'
    data_ver = '2'
    # =============================================================================
    # Hyperparameters
    # =============================================================================
    conv_L1 = 4
    kernel_size = 3
    pool_size = 2
    eta = 3e-4
    alpha = 1e-4
    weight_decay = 1e-5
    batch_size = 64
    epochs = 10
    optim_name = ''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, CNN_1D, optimizer):
        
        # load data attributes
        self.data_loader = ld.data_loader(choice=self.datatype, version=self.data_ver)
        self.train_data, self.valid_data, self.test_data = self.data_loader.load_shuffled()
        # self.n_features = self.data_loader.n_features
        self.n_features = 4
        # self.n_classes = self.data_loader.n_classes
        self.n_classes = 6
        # self.seq_length = self.data_loader.seq_length
        self.seq_length = 180

        self.model = CNN_1D(n_features=self.n_features, n_classes=self.n_classes, seq_length=self.seq_length,
                            conv_L1=self.conv_L1,  kernel_size=self.kernel_size, pool_size=self.pool_size).to(self.device)

        match optimizer:
            case 'Adam':
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=self.eta, weight_decay=self.alpha)
            case 'SGD':
                self.optimizer = optim.SGD(self.model.parameters(
                ), lr=self.eta, weight_decay=self.weight_decay, momentum=self.alpha)
            case 'RMSprop':
                self.optimizer = optim.RMSprop(self.model.parameters(
                ), lr=self.eta, weight_decay=self.weight_decay, momentum=self.alpha)

        self.optim_name = optimizer
        self.criterion = nn.CrossEntropyLoss()


    
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

    def fit(self, validate):
        # helper variables
        train_print_steps = 800
        val_print_steps = 50
        # plot_steps = 200

        train_loss = 0
        valid_loss = 0

        # load data
        train_batches = self.data_loader.load_batch_shuffled(data_list=self.train_data, batch_size=self.batch_size)
        valid_batches = self.data_loader.load_batch_shuffled(data_list=self.valid_data, batch_size=self.batch_size)

        # per epoch, there will be a training loop through all the trianing data followed by a validation loop through all the validation data
        for epoch in range(self.epochs):

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
                input_tensor, target_tensor = self.data_loader.batch_seq(batch)
                input_tensor = input_tensor.float()
                output = self.model(input_tensor.to(self.device))
                # convert target tensor to LongTensor for compatibility
                target_tensor = target_tensor.type(torch.LongTensor).to(self.device)

                # backpropagation
                self.optimizer.zero_grad()
                loss = self.criterion(output, target_tensor)
                loss.backward()
                self.optimizer.step()

                outputs = torch.argmax(output, dim=1)
                correct += (outputs == target_tensor).sum().item()

                if (i + 1) % (train_print_steps) == 0:
                    print(
                        f'Epoch {epoch}, batch number: {i + 1}, training loss = {loss}')
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
                        input_tensor, target_tensor = self.data_loader.batch_cnn_seq(
                            batch)
                        input_tensor = input_tensor.float()
                        v_output = self.model(input_tensor.to(self.device))
                        # convert target tensor to LongTensor for compatibility
                        target_tensor = target_tensor.type(
                            torch.LongTensor).to(self.device)

                        # calculate loss and valid_loss
                        v_loss = self.criterion(v_output, target_tensor)
                        v_outputs = torch.argmax(v_output, dim=1)
                        v_correct += (v_outputs == target_tensor).sum().item()

                        if (i + 1) % (val_print_steps) == 0:
                            print(
                                f'Epoch {epoch}, validation batch number: {i + 1}, validation loss = {v_loss}')
                            valid_loss_counter += 1
                            valid_loss += v_loss.item()

                val_accuracy = 100 * \
                    (v_correct / (len(valid_batches) * self.batch_size))
                self.validation_accuracies.append(val_accuracy)
                self.validation_losses.append(valid_loss/valid_loss_counter)
                print("========================================================================================================================================================== \n" +
                      f" ------------------> validation accuracy = {val_accuracy}, average validation loss = {valid_loss / valid_loss_counter} <------------------" +
                      "========================================================================================================================================================== \n")

                valid_loss = 0
                valid_loss_counter = 0

        # history
        self.history['training_accuracy'] += self.training_accuracies
        self.history['training_loss'] += self.training_losses
        self.history['validation_accuracy'] += self.validation_accuracies
        self.history['validation_loss'] += self.validation_losses

    # =============================================================================
    # method that tests model on test data
    # =============================================================================

    def predict(self):

        test_correct = 0
        test_loss_counter = 0
        test_loss = 0
        test_print_steps = 50
        test_batches = self.data_loader.load_batch_shuffled(
            data_list=self.test_data, batch_size=self.batch_size)

        for i, batch in enumerate(test_batches):
            self.model.eval()
            with torch.no_grad():
                # forward propagataion
                input_tensor, target_tensor = self.data_loader.batch_cnn_seq(
                    batch)
                input_tensor = input_tensor.float()
                test_output = self.model(input_tensor.to(self.device))
                # convert target tensor to LongTensor for compatibility
                target_tensor = target_tensor.type(
                    torch.LongTensor).to(self.device)

                # calculate loss and valid_loss
                t_loss = self.criterion(test_output, target_tensor)
                test_outputs = torch.argmax(test_output, dim=1)
                test_correct += (test_outputs == target_tensor).sum().item()

                if (i + 1) % (test_print_steps) == 0:
                    print(f'test batch number: {i + 1}, test loss = {t_loss}')
                    test_loss_counter += 1
                    test_loss += t_loss.item()

        test_accuracy = 100 * \
            (test_correct / (len(test_batches) * self.batch_size))
        self.test_losses.append(test_loss/test_loss_counter)
        self.test_accuracies.append(test_accuracy)
        print("========================================================================================================================================================== \n" +
              f" ------------------> test accuracy = {test_accuracy}, average test loss = {test_loss / test_loss_counter} <------------------" +
              "========================================================================================================================================================== \n")

        test_loss = 0
        test_loss_counter = 0

        # history
        self.history['test_accuracy'] += self.test_accuracies
        self.history['test_loss'] += self.test_losses

    # =============================================================================
    # method to save model to state_dict
    # =============================================================================

    def save_model(self, version_number):
        self.version_number = version_number
        torch.save(self.model.state_dict(),
                   f'saved_models/CNN_1D_v{self.version_number}.pt')
        print(f'CNN_1D_v{version_number} state_dict successfully saved')
        self.save_history(version_number)

    # =============================================================================
    # method to load model from state_dict
    # =============================================================================
    def load_model(self, version_number):
        self.version_number = version_number
        self.model.load_state_dict(torch.load(
            f'saved_models/CNN_1D_v{version_number}.pt'))
        print(
            f'CNN_1D_v{self.version_number} state dictionary successfully loaded')
        self.load_history(version_number)

    # =============================================================================
    # method to print params of model
    # =============================================================================

    def print_params(self):
        params = self.model.parameters()
        for p in params:
            print(p)

    # =============================================================================
    # returns history
    # =============================================================================
    def return_history(self):
        return self.history

    # =============================================================================
    # method that saves history to a pkl file
    # =============================================================================
    def save_history(self, version_number):
        with open(f'pkl/CNN_1D_v{version_number}_history.pkl', 'wb') as f:
            pickle.dump(self.history, f)
            print('CNN_1D_v{version_number} history saved')

    # =============================================================================
    # method that saves history to a pkl file
    # =============================================================================

    def load_history(self, version_number):
        with open(f'pkl/CNN_1D_v{version_number}_history.pkl', 'rb') as f:
            self.history = pickle.load(f)
            print('CNN_1D_v{version_number} history loaded')

    # =============================================================================
    # plot given metric
    # =============================================================================

    def plot(self, metric):
        match metric:
            case 'training_accuracy':
                stats = self.history['training_accuracy']
                plt.figure()
                plt.plot(stats)
                plt.title('Training accuracy')
                plt.xlabel('epochs')
                plt.ylabel('accuracy %')
                plt.show()
            case 'training_loss':
                stats = self.history['training_loss']
                plt.figure()
                plt.plot(stats)
                plt.title('Training loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.show()
            case 'validation_accuracy':
                stats = self.history['validation_accuracy']
                plt.figure()
                plt.plot(stats)
                plt.title('Validation accuracy')
                plt.xlabel('epochs')
                plt.ylabel('accuracy %')
                plt.show()
            case 'validation_loss':
                stats = self.history['validation_loss']
                plt.figure()
                plt.plot(stats)
                plt.title('Validation loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.show()
            case 'test_accuracy':
                stats = self.history['test_accuracy']
                plt.figure()
                plt.plot(stats)
                plt.title('Test accuracy')
                plt.xlabel('epochs')
                plt.ylabel('accuracy %')
                plt.show()
            case 'test_loss':
                stats = self.history['test_loss']
                plt.figure()
                plt.plot(stats)
                plt.title('Test loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.show()

    # =============================================================================
    # plot given metric
    # =============================================================================

    def print_summary(self):
        print(f'\nModel: CNN_1D_v{self.version_number} -> Hyperparamters: \n'
              f'Learnig rate = {self.eta} \nOptimiser = {self.optim_name} \nLoss = CrossEntropyLoss \n'
              f'conv_L1 = {self.conv_L1} \nkernel_size = {self.kernel_size} \npool_size = {self.pool_size} \n'
              f'Batch size = {self.batch_size} \nEpochs = {self.epochs} \nModel structure \n{self.model.eval()}'
              f'\nData: {self.datatype}, version {self.data_ver}, intervals of 8 minutes \nSequence length = {self.seq_length} \nBatch size = {self.batch_size}'
              )
        
        print(f'=====================================================================================================================\n'
              f'|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |'
              f'=====================================================================================================================\n'
              f'|        {"{:.4f}".format(round(model.history["training_accuracy"][-1], 4))}       |     {"{:.3f}".format(round(model.history["training_loss"][-1], 3))}      |        {"{:.3f}".format(round(model.history["validation_accuracy"][-1], 3))}         |      {"{:.4f}".format(round(model.history["validation_loss"][-1], 4))}       |      {"{:.3f}".format(round(model.history["test_accuracy"][0], 3))}     |    {"{:.4f}".format(round(model.history["test_loss"][0], 4))}   |'
              f'=====================================================================================================================')

# =============================================================================
# instantiate model and wrapper then train and save
# =============================================================================
model = CNN_1D_wrapper(CNN_1D, optimizer='Adam')
print(model.seq_length)
model.fit(validate=True)
# model.predict()
# print(model.history)
# model.save_model('8')
# model.print_summary()
# model.plot('training_accuracy')


# =============================================================================
# instantiate model and wrapper then load
# =============================================================================
# model = CNN_1D_wrapper(CNN_1D, optimizer='Adam')
# model.load_model('8')
# model.predict()
# model.print_summary()
# model.plot('validation_loss')
# num = '{:.3f}'.format(round(model.history["test_accuracy"][0], 3))
