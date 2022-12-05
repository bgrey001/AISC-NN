#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:57:37 2022

@author: benedict

Gated recurrent unit network implementation with PyTorch (GRU_v1)


"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader
import AIS_loader as data_module


# =============================================================================
# model class inherits from torch Module class
# =============================================================================
class GRU(nn.Module):

    # =============================================================================
    # class attributes
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================================================================
    # constructor
    # =============================================================================
    def __init__(self, n_features, hidden_dim, n_layers, n_classes, seq_length, batch_size):
        super(GRU, self).__init__()
        
        self.dropout_prob = 0
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim  
        self.batch_size = batch_size
        
        self.gru = nn.GRU(input_size=n_features, 
                          hidden_size=hidden_dim, 
                          num_layers=n_layers, 
                          batch_first=True, 
                          dropout=self.dropout_prob, 
                          bidirectional=True)
        
        # flatten and prediction layers
        self.fc_1 = nn.Linear(hidden_dim * 2, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    # =============================================================================
    # forward propagation method
    # =============================================================================
    def forward(self, input_x):
        
        h0 = torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_dim).to(self.device) # init hidden state, as it can't exist before the first forward prop
        input_x, _ = self.gru(input_x, h0)
        
        # prediction layers
        input_x = F.relu(self.fc_1(input_x))
        input_x = F.relu(self.fc_2(input_x))
        input_x = F.relu(self.fc_3(input_x))
        output = self.softmax(input_x)
        return output[:, -1]


    def init_hidden(self, input_x):
        return torch.zeros(self.n_layers, input_x.size(0), self.hidden_dim).requires_grad()
    
    
# =============================================================================
# wrapper class for an instance of the GRU_RNN model
# =============================================================================
class GRU_wrapper():

    # =============================================================================
    # Class attributes
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # =============================================================================
    # Data attributes
    # =============================================================================
    dataset = 'varying' # being padded in the AIS_loader class
    datatype = 'padded'
    data_ver = '3'
    shuffle = True
    # =============================================================================
    # Hyperparameters
    # =============================================================================
    n_layers = 2
    hidden_dim = 64
    eta = 3e-4
    alpha = 1e-4
    weight_decay = 1e-5
    batch_size = 64
    epochs = 5
    optim_name = ''
    # =============================================================================
    # constructor method
    # =============================================================================
    def __init__(self, GRU, optimizer, combine=False):
        
        # instantiate class members
        self.version_number = 0

        self.history = {'training_accuracy': [], 'training_loss': [], 'validation_accuracy': [], 'validation_loss': [], 'test_accuracy': [], 'test_loss': []}
        self.training_losses, self.validation_losses, self.test_losses, self.training_accuracies, self.validation_accuracies, self.test_accuracies = ([] for i in range(6))
        self.class_accuracies = None
        
        
        # instantiate dataset class objects
        if combine:
            self.train_data = data_module.AIS_loader(choice=self.dataset, split='train', version=self.data_ver, split_2='valid')
        else:
            self.train_data = data_module.AIS_loader(choice=self.dataset, split='train', version=self.data_ver)
            self.valid_data = data_module.AIS_loader(choice=self.dataset, split='valid', version=self.data_ver)
        self.test_data = data_module.AIS_loader(choice=self.dataset, split='test', version=self.data_ver)
        
        self.n_features = self.train_data.n_features
        self.n_classes = self.train_data.n_classes
        self.seq_length = self.train_data.seq_length

        self.model = GRU(n_features=self.n_features, 
                         hidden_dim=self.hidden_dim, 
                         n_layers=self.n_layers, 
                         n_classes=self.n_classes, 
                         seq_length=self.seq_length,
                         batch_size=self.batch_size).to(self.device)

        match optimizer:
            case 'AdamW':
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.eta, weight_decay=self.alpha)
            case 'SGD':
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.eta, weight_decay=self.weight_decay, momentum=self.alpha)
            case 'RMSprop':
                self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.eta, weight_decay=self.weight_decay, momentum=self.alpha)

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
    def fit(self, validate, epochs=None):
        if epochs == None:
            epochs = self.epochs
        else:
            self.epochs = epochs
        
        train_print_steps = 200
        val_print_steps = plot_steps = 20
        train_loss = valid_loss = 0

        # instantiate DataLoader
        train_generator = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.train_data.GRU_collate, drop_last=True)
        if validate:
            valid_generator = DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.valid_data.GRU_collate, drop_last=True)

        for epoch in range(epochs):

            correct = 0
            v_correct = 0
            train_loss_counter = 0
            valid_loss_counter = 0

            # =============================================================================
            # training loop
            # =============================================================================
            index = 0
            for features, labels, lengths in train_generator:
                # with torch.no_grad():
                # self.model.eval()
                self.model.train()
                features, labels = features.to(self.device), labels.to(self.device) # transfer to GPU
                # forward propagation
                output = self.model(features)
                # output = output[:, -1]
                # backpropagation
                self.optimizer.zero_grad()
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                # outputs = torch.zeros(labels.size()[0]).int().to(self.device)
                outputs = torch.argmax(output, dim=1)
                
                correct += (outputs == labels).sum().item()
            
                if index == 0 and epoch == 0:
                    first_accuracy = (100 * (correct / labels.size()[0]))
                    print(f'Initial accuracy = {first_accuracy}')
                    
                if (index + 1) % plot_steps == 0:
                    # metric = MulticlassAccuracy(num_classes=self.n_classes, average=None).to(self.device)
                    # print(metric(outputs, labels))
                    if (index + 1) % train_print_steps == 0:
                        print(f'Epoch {epoch}, batch number: {index + 1}, training loss = {loss}')
                    train_loss_counter += 1
                    train_loss += loss.item()
                            
                index += 1
                
            train_accuracy = 100 * (correct / (len(train_generator) * self.batch_size))
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
                v_index = 0
                for valid_features, valid_labels, lengths in valid_generator:
                    self.model.eval()
                    valid_features, valid_labels = valid_features.to(self.device), valid_labels.to(self.device)
                    with torch.no_grad():
                        # forward propagataion
                        valid_output = self.model(valid_features)
                        # calculate loss and valid_loss
                        v_loss = self.criterion(valid_output, valid_labels)
                        valid_outputs = torch.argmax(valid_output, dim=1)
                        # valid_outputs = torch.zeros(valid_labels.size()[0]).int().to(self.device)
                        v_correct += (valid_outputs == valid_labels).sum().item()

                        if (v_index + 1) % plot_steps == 0:
                            # metric = MulticlassAccuracy(num_classes=self.n_classes, average=None).to(self.device)
                            # print(metric(valid_outputs, valid_labels))
                            if (v_index + 1) % val_print_steps == 0:
                                print(f'Epoch {epoch}, validation batch number: {v_index + 1}, validation loss = {v_loss}')
                            valid_loss_counter += 1
                            valid_loss += v_loss.item()
                            
                    v_index += 1
                val_accuracy = 100 * (v_correct / (len(valid_generator) * self.batch_size))
                self.validation_accuracies.append(val_accuracy)
                self.validation_losses.append(valid_loss/valid_loss_counter)
                print("========================================================================================================================================================== \n" +
                      f" ------------------> validation accuracy = {val_accuracy}, average validation loss = {valid_loss / valid_loss_counter} <------------------" +
                      "========================================================================================================================================================== \n")

                valid_loss = 0
                valid_loss_counter = 0
            else:
                self.predict()
                
            if epoch % 2 == 0:
                self.confusion_matrix()
                print(f'Class accuracies: {self.class_accuracies}\n')

        # history
        self.history['training_accuracy'] += self.training_accuracies
        self.history['training_loss'] += self.training_losses
        self.history['validation_accuracy'] += self.validation_accuracies
        self.history['validation_loss'] += self.validation_losses

    # =============================================================================
    # method that tests model on test data
    # =============================================================================
    def predict(self):

        test_correct = test_loss_counter = test_loss = 0
        test_print_steps = 20
        
        test_generator = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.test_data.GRU_collate, drop_last=True)
        test_index = 0
        for test_features, test_labels, lengths in test_generator:
            self.model.eval()
            with torch.no_grad():
                test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)
                test_output = self.model(test_features)
                
                # calculate loss and valid_loss
                t_loss = self.criterion(test_output, test_labels)
                test_outputs = torch.argmax(test_output, dim=1)
                # test_outputs = torch.zeros(test_labels.size()[0]).int().to(self.device)
                test_correct += (test_outputs == test_labels).sum().item()
            
                if (test_index + 1) % (test_print_steps) == 0:
                    print(f'test batch number: {test_index + 1}, test loss = {t_loss}')
                    test_loss_counter += 1
                    test_loss += t_loss.item()
            test_index += 1

        test_accuracy = 100 * (test_correct / (len(test_generator) * self.batch_size))
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
    # method to build confusion matrix
    # =============================================================================
    def confusion_matrix(self):
        class_list = ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers'] # just for visual reference
        test_generator = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.train_data.GRU_collate, drop_last=True)
        confusion_matrix = torch.zeros(self.n_classes, self.n_classes)
        with torch.no_grad():
            self.model.eval()
            for test_features, test_labels, lengths in test_generator:
                test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)
                test_output = self.model(test_features)
                preds = torch.argmax(test_output, dim=1)
                # preds = torch.zeros(test_labels.size()[0]).int().to(self.device)
                for t, p in zip(test_labels.view(-1), preds):
                    confusion_matrix[t.long(), p.long()] += 1
        self.class_accuracies = (confusion_matrix.diag()/confusion_matrix.sum(1)).numpy()
    
    # =============================================================================
    # method to save model to state_dict
    # =============================================================================

    def save_model(self, version_number, init=False):
        self.version_number = version_number
        if init:
            torch.save(self.model.state_dict(), f'saved_models/init_param_models/GRU_v{version_number}.pt')
        else:
            torch.save(self.model.state_dict(), f'saved_models/GRU_v{version_number}.pt')
            
        print(f'GRU_v{version_number} state_dict successfully saved')
        self.save_history(version_number, init)

    # =============================================================================
    # method to load model from state_dict
    # =============================================================================
    def load_model(self, version_number, init=False):
        self.version_number = version_number
        if init:
            self.model.load_state_dict(torch.load(f'saved_models/init_param_models/GRU_v{version_number}.pt'))
        else:
            self.model.load_state_dict(torch.load(f'saved_models/GRU_v{version_number}.pt'))
        print(f'GRU_v{version_number} state dictionary successfully loaded')
        self.load_history(version_number, init)

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
    def save_history(self, version_number, init=False):
        if init:
            with open(f'saved_models/history/init_histories/GRU_v{version_number}_history.pkl', 'wb') as f:
                pickle.dump(self.history, f)
        else:
            with open(f'saved_models/history/GRU_v{version_number}_history.pkl', 'wb') as f:
                pickle.dump(self.history, f)
        print(f'GRU_v{version_number} history saved')

    # =============================================================================
    # method that saves history to a pkl file
    # =============================================================================

    def load_history(self, version_number, init=False):
        if init:
            with open(f'saved_models/history/init_histories/GRU_v{version_number}_history.pkl', 'rb') as f:
                self.history = pickle.load(f)
        else:
            with open(f'saved_models/history/GRU_v{version_number}_history.pkl', 'rb') as f:
                self.history = pickle.load(f)
        print(f'GRU_v{version_number} history loaded')

    # =============================================================================
    # plot given metric
    # =============================================================================

    def plot(self, metric):
        match metric:
            case 'training_accuracy':
                y = np.array(self.history['training_accuracy'])
                x = np.arange(len(y)) # number of epochs
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.plot(x_, y_)
                plt.title('Training accuracy')
                plt.xlabel('epochs')
                plt.ylabel('accuracy %')
                plt.show()
            case 'training_loss':
                y = np.array(self.history['training_loss'])
                x = np.arange(len(y)) # number of epochs
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.plot(x_, y_)
                plt.title('Training loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.show()
            case 'validation_accuracy':
                y = np.array(self.history['validation_accuracy'])
                x = np.arange(len(y)) # number of epochs
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.plot(x_, y_)
                plt.title('Validation accuracy')
                plt.xlabel('epochs')
                plt.ylabel('accuracy %')
                plt.show()
            case 'validation_loss':
                y = np.array(self.history['validation_loss'])
                x = np.arange(len(y)) # number of epochs
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.plot(x_, y_)
                plt.title('Validation loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.show()
            case 'test_accuracy':
                y = np.array(self.history['test_accuracy'])
                x = np.arange(len(y)) # number of epochs
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.plot(x_, y_)
                plt.title('Test accuracy')
                plt.xlabel('epochs')
                plt.ylabel('accuracy %')
                plt.show()
            case 'test_loss':
                y = np.array(self.history['test_loss'])
                x = np.arange(len(y)) # number of epochs
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.plot(x_, y_)
                plt.title('Test loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.show()
                
                

    # =============================================================================
    # plot given metric
    # =============================================================================

    def print_summary(self):
        self.confusion_matrix()
        print(f'\nModel: GRU_v{self.version_number} -> Hyperparamters: \n'
              f'Learnig rate = {self.eta} \nOptimiser = {self.optim_name} \nLoss = CrossEntropyLoss \n'
              f'conv_l1 = {self.conv_l1} \nkernel_size = {self.kernel_size} \npool_size = {self.pool_size} \n'
              f'Batch size = {self.batch_size} \nEpochs = {self.epochs} \nModel structure \n{self.model.eval()}'
              f'\nData: {self.datatype}, v{self.data_ver}, varying intervals \nSequence length = {self.seq_length} \nBatch size = {self.batch_size} \nShuffled = {self.shuffle}'
              )
        
        print('\nMetric table')
        
        print(f'=====================================================================================================================\n'
              f'|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |'
              f'=====================================================================================================================\n'
              f'|        {"{:.3f}".format(round(self.history["training_accuracy"][-1], 4))}%      |      {"{:.3f}".format(round(self.history["training_loss"][-1], 3))}      |        {"{:.3f}".format(round(self.history["validation_accuracy"][-1], 3))}%        |       {"{:.3f}".format(round(self.history["validation_loss"][-1], 4))}       |      {"{:.3f}".format(round(self.history["test_accuracy"][-1], 3))}%    |     {"{:.3f}".format(round(self.history["test_loss"][-1], 4))}   |'
              f'=====================================================================================================================')
        
        print('\nClass accuracy table')

        print(f'=====================================================================================================================\n'
              f'|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |'
              f'=====================================================================================================================\n'
              f'|           {"{:1d}".format(round(100 * self.class_accuracies[0]))}%          |        {"{:1d}".format(round(100 * self.class_accuracies[1]))}%       |        {"{:1d}".format(round(100 * self.class_accuracies[2]))}%        |       {"{:1d}".format(round(100 * self.class_accuracies[3]))}%        |        {"{:1d}".format(round(100 * self.class_accuracies[4]))}%     |       {"{:1d}".format(round(100 * self.class_accuracies[5]))}%     |'
              f'=====================================================================================================================')



# =============================================================================
# instantiate K models and select the model with the highest validation accuracy
# =============================================================================
def init_params(K):
    print(f'Beginning random initalisation with {K} different models')
    records = {'index': None, 'highest_accuracy': 0}
    # models = []
    for k in range(K):
        model = GRU_wrapper(GRU, optimizer='AdamW')
        print(f'MODEL {k + 1} -------------------------------->')
        model.fit(validate=True, epochs=3)
        # print(models[k].history['validation_accuracy'])
        if max(model.history['validation_accuracy']) > records['highest_accuracy']:
            records['index'] = k + 1
            print(f'New highest record: model {k + 1}')
            records['highest_accuracy'] = max(model.history['validation_accuracy'])
        model.save_model(k + 1, init=True)
        del model
        
    # save highest index 
    with open('saved_models/history/init_histories/highest_idx.pkl', 'wb') as f:
        pickle.dump(records['index'], f)
        print(f"Highest_idx = {records['index']}, saved successfully")
    
    
# =============================================================================
# run random initalisation and then load the model with the highest validation accuracy for more training
# =============================================================================
# init_params(K=20)

    
# =============================================================================
# load the parameters of the model with the highest validation accuracy
# =============================================================================
def load_highest_model(model):
    with open('saved_models/history/init_histories/highest_idx.pkl', 'rb') as f:
        highest_idx = pickle.load(f)
        print(f"Highest_idx = {highest_idx}, loaded successfully")
    model.load_model(highest_idx, init=True)
    
    
# =============================================================================
# load the best randomly initialised network parameters for further training
# =============================================================================

model = GRU_wrapper(GRU, optimizer='AdamW', combine=False)
# load_highest_model(model)
# print(model.history)
# model.load_model(2)
model.fit(validate=True, epochs=25)
model.predict()
model.print_summary()
# model.fit(validate=True, epochs=30)
# model.print_summary()

# model.confusion_matrix()
# model.save_model(2)
# model.predict()



# model.plot('training_accuracy')
# model.plot('validation_accuracy')
# model.plot('training_loss')
# model.plot('validation_loss')

# model.plot('test_accuracy')




