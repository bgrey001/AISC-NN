#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:57:37 2022
@author: Benedict Grey

Gated recurrent unit network implementation with PyTorch (GRU_v1)

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from pycm import ConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.interpolate import make_interp_spline
from datetime import datetime

import AIS_loader as data_module

sns.set_style("darkgrid") 

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
    def __init__(self, n_features, hidden_dim, n_layers, n_classes, seq_length, batch_size, bidirectional):
        super(GRU, self).__init__()
        
        bi_dim = 1
        if bidirectional:
            bi_dim = 2
        
        self.bi_dim = bi_dim
        
        self.dropout_prob = 0
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim  
        self.batch_size = batch_size
        
        self.gru = nn.GRU(input_size=n_features, 
                          hidden_size=hidden_dim, 
                          num_layers=n_layers, 
                          batch_first=True, 
                          dropout=self.dropout_prob, 
                          bidirectional=bidirectional)
        
        # flatten and prediction layers
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_dim * bi_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, n_classes)

    # =============================================================================
    # forward propagation method
    # =============================================================================
    def forward(self, input_x):
        
        
        h0 = torch.zeros(self.n_layers * self.bi_dim, self.batch_size, self.hidden_dim).to(self.device) # init hidden state, as it can't exist before the first forward prop
        gru_out, hidden = self.gru(input_x, h0)
        
        
        """
        We are taking the first and final predictions of each sequence and concatenating them. 
        This represents the final outputs of the GRU for forward and reverse directions:
            gru_out[:, -1, :self.hidden_dim] is the last hidden state of the forward pass
            gru_out[:, 0, self.hidden_dim:]  is the last hidden state of the backward pass
            fc_in = torch.cat((gru_out[:, -1, :self.hidden_dim], gru_out[:, 0, self.hidden_dim:]), dim=1)
        
        The following method produces the same results (have opted for the first method as involves less data transformations):
            hidden = hidden.view(self.n_layers, self.bi_dim, self.batch_size, self.hidden_dim)
            hidden = hidden[-1]
            hidden_forward, hidden_backward = hidden[0], hidden[1]
            fc_in = torch.cat((hidden_forward, hidden_backward), dim = 1)
        """

        fc_in = torch.cat((gru_out[:, -1, :self.hidden_dim], gru_out[:, 0, self.hidden_dim:]), dim=1)

        # classification layers
        fc_in = self.relu(self.fc_1(fc_in))
        fc_in = self.relu(self.fc_2(fc_in))
        output = self.fc_3(fc_in)
        return output

 
    
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
    data_ver = '3'
    shuffle = True
    # =============================================================================
    # Hyperparameters
    # =============================================================================
    eta = 3e-4
    alpha = 1e-4
    optim_name = ''
    # =============================================================================
    # constructor method
    # =============================================================================
    def __init__(self, GRU, dataset, n_units, hidden_dim, optimizer, bidirectional, batch_size, combine=False):
        
        # init class members
        self.dataset = dataset
        self.version_number = 0
        self.batch_size = batch_size
        
        self.history = {'training_accuracy': [], 
                        'training_loss': [], 
                        'validation_accuracy': [], 
                        'validation_loss': [], 
                        'test_accuracy': [], 
                        'test_loss': [],
                        
                        'confusion_matrix': None,
                        'class_precisions': None,
                        'class_recalls': None,
                        'class_F1_scores': None,
                        'class_supports': None}
        
        self.training_losses, self.validation_losses, self.test_losses, self.training_accuracies, self.validation_accuracies, self.test_accuracies = ([] for i in range(6))
        self.class_accuracies = None
        self.epochs = len(self.history['training_accuracy'])

        # init dataset class objects
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
                         hidden_dim=hidden_dim, 
                         n_layers=n_units, 
                         n_classes=self.n_classes, 
                         seq_length=self.seq_length,
                         batch_size=batch_size,
                         bidirectional=bidirectional).to(self.device)

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
            start_time = datetime.now()  
            aggregate_correct = 0
            v_correct = 0
            train_loss_counter = 0
            valid_loss_counter = 0

            # =============================================================================
            # training loop
            # =============================================================================
            index = 0
            for features, labels, lengths in train_generator:
                # with torch.no_grad():
                self.model.train()
                features, labels = features.to(self.device), labels.to(self.device) # transfer to GPU
                # forward propagation
                output = self.model(features)

                # backpropagation
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                outputs = torch.argmax(output, dim=1)
                aggregate_correct += (((outputs == labels).sum().item()) / len(labels)) * 100
            
                if index == 0 and epoch == 0:
                    first_accuracy = aggregate_correct
                    print(f'Initial accuracy = {first_accuracy}')
                    
                if (index + 1) % plot_steps == 0:
                    # metric = MulticlassAccuracy(num_classes=self.n_classes, average=None).to(self.device)
                    # print(metric(outputs, labels))
                    if (index + 1) % train_print_steps == 0:
                        print(f'Epoch {epoch}, batch number: {index + 1}, training loss = {loss}')
                    train_loss_counter += 1
                    train_loss += loss.item()
                            
                index += 1
                
            train_accuracy = aggregate_correct / (len(train_generator))
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
                        v_correct += ((valid_outputs == valid_labels).sum().item() / len(valid_labels)) * 100

                        if (v_index + 1) % plot_steps == 0:
                            # metric = MulticlassAccuracy(num_classes=self.n_classes, average=None).to(self.device)
                            # print(metric(valid_outputs, valid_labels))
                            if (v_index + 1) % val_print_steps == 0:
                                print(f'Epoch {epoch}, validation batch number: {v_index + 1}, validation loss = {v_loss}')
                            valid_loss_counter += 1
                            valid_loss += v_loss.item()
                            
                    v_index += 1
                val_accuracy = v_correct / (len(valid_generator))
                self.validation_accuracies.append(val_accuracy)
                self.validation_losses.append(valid_loss/valid_loss_counter)
                print("========================================================================================================================================================== \n" +
                      f" ------------------> validation accuracy = {val_accuracy}, average validation loss = {valid_loss / valid_loss_counter} <------------------" +
                      "========================================================================================================================================================== \n")

                valid_loss = 0
                valid_loss_counter = 0
            else:
                self.predict()
                
            self.confusion_matrix(valid=validate)
            print(f'Class F1-scores: {self.history["class_F1_scores"]}\n')
            end_time = datetime.now()
            print(f'Epoch duration: {(end_time - start_time)}')

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
                test_correct += ((test_outputs == test_labels).sum().item() / len(test_labels)) * 100 

                if (test_index + 1) % (test_print_steps) == 0:
                    print(f'test batch number: {test_index + 1}, test loss = {t_loss}')
                    test_loss_counter += 1
                    test_loss += t_loss.item()
            test_index += 1

        test_accuracy = test_correct / (len(test_generator))
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
    def confusion_matrix(self, valid=False, save_fig=False, print_confmat=False):
        
        test_correct = 0
        predicted, labels = [], []
        n_correct = 0
        
        class_list = ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers'] # just for visual reference
        if valid:
            test_generator = DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.test_data.GRU_collate, drop_last=True)
        else:
            test_generator = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.test_data.GRU_collate, drop_last=True)
       
        with torch.no_grad():
            self.model.eval()
            for test_features, test_labels, lengths in test_generator:
                test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)
                test_output = self.model(test_features)
                preds = torch.argmax(test_output, dim=1)

                # confmat variables
                predicted.append(preds.cpu().detach().numpy())
                labels.append(test_labels.cpu().detach().numpy())

                n_correct += ((preds == test_labels).sum().item() / len(test_labels)) * 100 
                test_correct += (preds == test_labels).sum().item()
        
        predicted = np.concatenate(predicted).ravel().tolist()
        labels = np.concatenate(labels).ravel().tolist()
    
        confmat = ConfusionMatrix(actual_vector=labels, predict_vector=predicted)
        
        if print_confmat:
            confmat.print_matrix()
            confmat.stat(summary=True)
        
        if save_fig:
            confmat.plot(cmap=plt.cm.Reds,number_label=True,plot_lib="matplotlib")
            plt.savefig(f'GRU_v{self.version_number}.png', dpi=300)
            plt.savefig(f'../../plots/GRU/v{self.version_number}/confmat_GRU_v{self.version_number}.png', dpi=300)
        
        self.history['confusion_matrix'] = confmat
        self.history['class_precisions'] = confmat.class_stat['PPV']
        self.history['class_recalls'] = confmat.class_stat['TPR']
        self.history['class_F1_scores'] = confmat.class_stat['F1']
        self.history['class_supports'] = confmat.class_stat['P']
        

    # =============================================================================
    # methdo to prune (probably low magnitude) weights to improve generalisation
    # =============================================================================
    def prune_weights(self, amount):
        parameters_to_prune = (
            # (self.model.gru, 'all_weights'),
            (self.model.fc_1, 'weight'),
            (self.model.fc_2, 'weight'),
            (self.model.fc_3, 'weight'),
        )

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        
        
    # =============================================================================
    # method returns total parameters in the network
    # =============================================================================
    def total_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        

    # =============================================================================
    # method to save model to state_dict
    # =============================================================================
    def save_model(self, version_number, init=False):
        self.version_number = version_number
        if init:
            torch.save(self.model.state_dict(), f'saved_models/init_param_models/GRU_v{version_number}.pt')
        elif self.dataset == 'varying':
            torch.save(self.model.state_dict(), f'saved_models/zero_padded/GRU_v{version_number}.pt')
        elif self.dataset == 'linear_interp':
            torch.save(self.model.state_dict(), f'saved_models/linear_interp/GRU_v{version_number}.pt')
            
        print(f'GRU_v{version_number} state_dict successfully saved')
        self.save_history(version_number, init)

    # =============================================================================
    # method to load model from state_dict
    # =============================================================================
    def load_model(self, version_number, init=False):
        self.version_number = version_number
        if init:
            self.model.load_state_dict(torch.load(f'saved_models/init_param_models/GRU_v{version_number}.pt'))
        elif self.dataset == 'varying':
            self.model.load_state_dict(torch.load(f'saved_models/zero_padded/GRU_v{version_number}.pt'))
        elif self.dataset == 'linear_interp':
            self.model.load_state_dict(torch.load(f'saved_models/linear_interp/GRU_v{version_number}.pt'))
            
        print(f'GRU_v{version_number} state dictionary successfully loaded')
        self.load_history(version_number, init)

    # =============================================================================
    # method to print params of model
    # =============================================================================
    def print_params(self):
        params = self.model.named_parameters()
        for name, param in params:
            # if param.requires_grad:
            print(name)
            print(param.data.shape)
            print()

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
        elif self.dataset == 'varying':
            with open(f'saved_models/history/zero_padded/GRU_v{version_number}_history.pkl', 'wb') as f:
                pickle.dump(self.history, f)
        elif self.dataset == 'linear_interp':
            with open(f'saved_models/history/linear_interp/GRU_v{version_number}_history.pkl', 'wb') as f:
                pickle.dump(self.history, f)
                
        print(f'GRU_v{version_number} history saved')

    # =============================================================================
    # method that saves history to a pkl file
    # =============================================================================
    def load_history(self, version_number, init=False):
        if init:
            with open(f'saved_models/history/init_histories/GRU_v{version_number}_history.pkl', 'rb') as f:
                self.history = pickle.load(f)
        elif self.dataset == 'varying':
            with open(f'saved_models/history/zero_padded/GRU_v{version_number}_history.pkl', 'rb') as f:
                self.history = pickle.load(f)
        elif self.dataset == 'linear_interp':
            with open(f'saved_models/history/linear_interp/GRU_v{version_number}_history.pkl', 'rb') as f:
                self.history = pickle.load(f)
        self.epochs = len(self.history['training_accuracy'])
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
                plt.grid(True)
                plt.plot(x_, y_, label='Training accuracy')
                plt.title('Training accuracy')
                plt.xlabel('epochs')
                plt.ylabel('accuracy %')
                plt.legend()
                plt.show()
            case 'training_loss':
                y = np.array(self.history['training_loss'])
                x = np.arange(len(y)) # number of epochs
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.grid(True)
                plt.plot(x_, y_)
                plt.title('Training loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.legend()
                plt.show()
            case 'validation_accuracy':
                y = np.array(self.history['validation_accuracy'])
                x = np.arange(len(y)) # number of epochs
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.grid(True)
                plt.plot(x_, y_)
                plt.title('Validation accuracy')
                plt.xlabel('epochs')
                plt.ylabel('accuracy %')
                plt.legend()
                plt.show()
            case 'validation_loss':
                y = np.array(self.history['validation_loss'])
                x = np.arange(len(y)) # number of epochs
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.grid(True)
                plt.plot(x_, y_)
                plt.title('Validation loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.legend()
                plt.show()
            case 'test_accuracy':
                y = np.array(self.history['test_accuracy'])
                x = np.arange(len(y)) # number of epochs
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.grid(True)
                plt.plot(x_, y_)
                plt.title('Test accuracy')
                plt.xlabel('epochs')
                plt.ylabel('accuracy %')
                plt.legend()
                plt.show()
            case 'test_loss':
                y = np.array(self.history['test_loss'])
                x = np.arange(len(y)) # number of epochsstrongest
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.grid(True)
                plt.plot(x_, y_)
                plt.title('Test loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.legend()
                plt.show()
            case 'accuracy':
                y = np.array(self.history['training_accuracy'])
                y_2 = np.array(self.history['validation_accuracy'])
                x = np.arange(len(y)) # number of epochs
                x_2 = np.arange(len(y_2)) # number of epochs
                spline = make_interp_spline(x, y)
                spline_2 = make_interp_spline(x_2, y_2)
                x_ = np.linspace(x.min(), x.max(), 500)
                x_2_ = np.linspace(x_2.min(), x_2.max(), 500)
                y_ = spline(x_)
                y_2_ = spline_2(x_2_)
                plt.grid(True)
                plt.plot(x_, y_, c='green', label='Training accuracy')
                plt.plot(x_, y_2_, c='orange', label='Validation accuracy')
                # plt.title('Accuracy')
                plt.xlabel('epochs')
                plt.ylabel('accuracy %')
                plt.legend()
                plt.show()
            case 'loss':
                y = np.array(self.history['training_loss'])
                y_2 = np.array(self.history['validation_loss'])
                x = np.arange(len(y)) # number of epochs
                x_2 = np.arange(len(y_2)) # number of epochs
                spline = make_interp_spline(x, y)
                spline_2 = make_interp_spline(x_2, y_2)
                x_ = np.linspace(x.min(), x.max(), 500)
                x_2_ = np.linspace(x_2.min(), x_2.max(), 500)
                y_ = spline(x_)
                y_2_ = spline_2(x_2_)
                plt.grid(True)
                plt.plot(x_, y_, c='green', label='Training loss')
                plt.plot(x_, y_2_, c='orange', label='Validation loss')
                # plt.title('Loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.legend()
                plt.show()
            
                
    # =============================================================================
    # plot given metric
    # =============================================================================

    def print_summary(self, print_cm=False):
        self.confusion_matrix()
        print(f'\nModel: GRU_v{self.version_number} -> Hyperparamters: \n'
              f'Learnig rate = {self.eta} \nOptimiser = {self.optim_name} \nLoss = CrossEntropyLoss \n'
              f'Batch size = {self.batch_size} \nEpochs = {self.epochs} \nModel structure: \n{self.model.eval()} \nTotal parameters = {self.total_params()}'
              f'\nData: {self.dataset}, v{self.data_ver} \nSequence length = {self.seq_length} \nBatch size = {self.batch_size} \nShuffled = {self.shuffle}'
              )
        
        print('\nMetric table')
        print(f'=====================================================================================================================\n'
              f'|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |\n'
              f'=====================================================================================================================\n'
              f'|        {"{:.3f}".format(round(self.history["training_accuracy"][-1], 4))}%      |      {"{:.3f}".format(round(self.history["training_loss"][-1], 3))}      |        {"{:.3f}".format(round(self.history["validation_accuracy"][-1], 3))}%        |       {"{:.3f}".format(round(self.history["validation_loss"][-1], 4))}       |      {"{:.3f}".format(round(self.history["test_accuracy"][-1], 3))}%    |     {"{:.3f}".format(round(self.history["test_loss"][-1], 4))}   |\n'
              f'=====================================================================================================================')
        
        print('\nClass F1-score table')
        print(f'=====================================================================================================================\n'
              f'|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |\n'
              f'=====================================================================================================================\n'
              f'|         {"{:.3f}".format((100 * self.history["class_F1_scores"][0]))}%        |      {"{:.3f}".format((100 * self.history["class_F1_scores"][1]))}%     |      {"{:.3f}".format((100 * self.history["class_F1_scores"][2]))}%      |     {"{:.3f}".format((100 * self.history["class_F1_scores"][3]))}%      |     {"{:.3f}".format((100 * self.history["class_F1_scores"][4]))}%    |    {"{:.3f}".format((100 * self.history["class_F1_scores"][5]))}%    |\n'
              f'=====================================================================================================================\n\n')
        if print_cm:
            self.confusion_matrix(print_confmat=(True))


# =============================================================================
# instantiate K models and select the model with the highest validation accuracy
# =============================================================================
def nonrandom_init(K, dataset):
    print(f'Beginning random initalisation with {K} different models')
    records = {'index': None, 'highest_accuracy': 0}
    # models = []
    for k in range(K):
        model = GRU_wrapper(GRU, dataset=dataset, n_units=2, hidden_dim=64, optimizer='AdamW', bidirectional=True, batch_size=64, combine=False)
        print(f'MODEL {k + 1} -------------------------------->')
        model.fit(validate=True, epochs=2)
        # print(models[k].history['validation_accuracy'])
        if max(model.history['validation_accuracy']) > records['highest_accuracy']:
            records['index'] = k + 1
            print(f'New highest record: model {k + 1}')
            records['highest_accuracy'] = max(model.history['validation_accuracy'])
        model.save_model(k + 1, init=True)
        del model
        
    # save highest index 
    with open('saved_models/history/init_histories/GRU_highest_idx.pkl', 'wb') as f:
        pickle.dump(records['index'], f)
        print(f"Highest_idx = {records['index']}, saved successfully")

    
# =============================================================================
# load the parameters of the model with the highest validation accuracy
# =============================================================================
def load_highest_model(model):
    with open('saved_models/history/init_histories/GRU_highest_idx.pkl', 'rb') as f:
        highest_idx = pickle.load(f)
        print(f"Highest_idx = {highest_idx}, loaded successfully")
    model.load_model(highest_idx, init=True)
    
    
    
if __name__ == "__main__":    
    # =============================================================================
    # load the best randomly initialised network parameters for further training
    # =============================================================================
    nonrand = False
    # current_dataset = 'linear_interp'
    current_dataset = 'varying'
    
    if nonrand:
        nonrandom_init(K=10, dataset=current_dataset)
        
    model = GRU_wrapper(GRU, 
                        dataset=current_dataset, 
                        n_units=2, 
                        hidden_dim=64, 
                        optimizer='AdamW', 
                        bidirectional=True, 
                        batch_size=64, 
                        combine=True)
    # load_highest_model(model)
    
    
    # =============================================================================
    # testing zone
    # =============================================================================
    model.load_model(11)
    model.fit(validate=False, epochs=10)
    # model.prune_weights(amount=0.2)
    model.predict()
    model.print_summary(print_cm=True)
    # model.confusion_matrix()
    model.save_model(12)
    
    # model.plot('training_accuracy')
    # model.plot('validation_accuracy')
    # model.plot('training_loss')
    # model.plot('validation_loss')
    model.plot('accuracy')
    model.plot('loss')





