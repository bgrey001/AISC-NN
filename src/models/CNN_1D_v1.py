#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:18:57 2022
@author: Benedict Grey

One dimensional convolutional neural network with residual blocks PyTorch implementation (CNN_1D_v1)

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from pycm import ConfusionMatrix
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader
from datetime import datetime

import AIS_loader as data_module

sns.set_style("darkgrid") 


# =============================================================================
# custom Residual block
# =============================================================================
class ResBlock(nn.Module):
    # =============================================================================
    # constructor
    # =============================================================================
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm1d(out_channels)
        self.conv_2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.batch_norm_2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        
    # =============================================================================
    # forward propagation method
    # =============================================================================
    def forward(self, input_x):
        residual = input_x # preserve input for identity connection
        
        input_x = self.conv_1(input_x) # conv block 1
        input_x = self.batch_norm_1(input_x)
        input_x = self.relu(input_x)
        
        input_x = self.conv_2(input_x) # conv block 2
        input_x = self.batch_norm_2(input_x)
        input_x = self.relu(input_x)
        
        input_x += residual # residual injected
        output = self.relu(input_x)
        return output
        

class CNN_1D(nn.Module):
    # =============================================================================
    # class attributes
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # =============================================================================
    # constructor
    # =============================================================================
    def __init__(self, n_features, n_classes, seq_length, conv_l1, kernel_size, pool_size):
        super(CNN_1D, self).__init__()
        # calculate channel sizes for the different convolution layers
        conv_l2 = 2 * conv_l1
        conv_l3 = conv_l2
        
        self.conv_1 = nn.Conv1d(in_channels=n_features, out_channels=conv_l1, kernel_size=kernel_size)
        self.batch_norm_1 = nn.BatchNorm1d(conv_l1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(pool_size)    

        self.res_block_1 = ResBlock(in_channels=conv_l1, out_channels=conv_l1, kernel_size=kernel_size)
        self.res_block_2 = ResBlock(in_channels=conv_l1, out_channels=conv_l1, kernel_size=kernel_size)
        self.avgpool = nn.AvgPool1d(pool_size)    
        
        self.conv_2 = nn.Conv1d(in_channels=conv_l1, out_channels=conv_l2, kernel_size=kernel_size)
        self.batch_norm_2 = nn.BatchNorm1d(conv_l2)
        
        # configure transformed dimensions of the input as it reaches the fully connected layer
        conv_l1_dim = math.floor((seq_length - (kernel_size - 1))/ pool_size)
        res_l1_dim = math.floor(conv_l1_dim / pool_size)
        conv_l2_dim = math.floor((res_l1_dim - (kernel_size - 1)) / pool_size)
        # res_l2_dim = math.floor(conv_l2_dim / pool_size)
        # conv_l3_dim = math.floor((res_l2_dim - (kernel_size - 1)) / pool_size)

        flat_size = conv_l2 * conv_l2_dim

        # flatten and prediction layers
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(flat_size, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, n_classes)
        


    # =============================================================================
    # forward propagation method
    # =============================================================================
    def forward(self, input_x):
        
        input_x = self.conv_1(input_x)
        input_x = self.batch_norm_1(input_x)
        input_x = self.maxpool(self.relu(input_x))

        input_x = self.res_block_1(input_x)
        input_x = self.res_block_2(input_x)

        input_x = self.avgpool(input_x)

        input_x = self.conv_2(input_x)
        input_x = self.batch_norm_2(input_x)
        input_x = self.maxpool(self.relu(input_x))
        input_x = self.flatten(input_x)
        input_x = F.relu(self.fc_1(input_x))
        input_x = F.relu(self.fc_2(input_x))
        input_x = F.relu(self.fc_3(input_x))

        return input_x

# =============================================================================
# wrapper class for an instance of the CNN_1D model
# =============================================================================
class CNN_1D_wrapper():
    # =============================================================================
    # Class attributes
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # =============================================================================
    # Data attributes
    # =============================================================================
    data_ver = '4'
    shuffle = True
    # =============================================================================
    # Hyperparameters
    # =============================================================================
    conv_l1 = 64
    kernel_size = 3
    pool_size = 2
    eta = 3e-4
    alpha = 1e-5
    optim_name = ''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, CNN_1D, dataset, optimizer, batch_size, combine=False):
        
        # init class members
        self.dataset = dataset
        self.datatype = dataset
        self.batch_size = batch_size
        self.version_number = 0

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

        self.model = CNN_1D(n_features=self.n_features, n_classes=self.n_classes, seq_length=self.seq_length, conv_l1=self.conv_l1,  kernel_size=self.kernel_size, pool_size=self.pool_size).to(self.device)

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

    def fit(self, validate, epochs):

        
        train_print_steps = 200
        val_print_steps = plot_steps = 20
        train_loss = valid_loss = 0

        # instantiate DataLoader
        train_generator = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.train_data.CNN_collate)
        # train_generator = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        if validate:
            valid_generator = DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.valid_data.CNN_collate)
            # valid_generator = DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=self.shuffle)

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
                # self.model.eval()
                self.model.train()
                features, labels = features.to(self.device), labels.to(self.device) # transfer to GPU
                # forward propagation
                output = self.model(features)
                # backpropagation
                self.optimizer.zero_grad()
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                outputs = torch.argmax(output, dim=1)
                aggregate_correct += (((outputs == labels).sum().item()) / len(labels)) * 100
            
                if index == 0 and epoch == 0:
                    first_accuracy = (100 * (aggregate_correct / len(labels)))
                    print(f'Initial accuracy = {first_accuracy}')
                    
                if (index + 1) % plot_steps == 0:
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
                        v_correct += ((valid_outputs == valid_labels).sum().item() / len(valid_labels)) * 100
                        if (v_index + 1) % plot_steps == 0:
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
        
        y_preds = []
        y_test = []

        test_correct = test_loss_counter = test_loss = 0
        test_print_steps = 20
        
        # for varying padded data
        test_generator = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.test_data.CNN_collate)
        test_index = 0
        
        for test_features, test_labels, lengths in test_generator:
            self.model.eval()
            with torch.no_grad():
                test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)
                test_output = self.model(test_features)
                
                # calculate loss and valid_loss
                t_loss = self.criterion(test_output, test_labels)
                test_outputs = torch.argmax(test_output, dim=1)
                y_preds.append(test_outputs)
                y_test.append(test_labels)
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
        return y_preds, y_test


    # =============================================================================
    # method to build confusion matrix
    # =============================================================================
    def confusion_matrix(self, valid=False, save_fig=False, print_confmat=False):
        
        test_correct = 0
        predicted, labels = [], []
        n_correct = 0
        
        class_list = ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers'] # just for visual reference
        if valid:
            test_generator = DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.test_data.CNN_collate)
        else:
            test_generator = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.train_data.CNN_collate)
            
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
            plt.savefig(f'../../plots/CNN_1D/linear_interp/ResBlock_v{self.version_number}/confmat_CNN_1D_v{self.version_number}.png', dpi=300)
        
        
        self.history['confusion_matrix'] = confmat
        self.history['class_precisions'] = confmat.class_stat['PPV']
        self.history['class_recalls'] = confmat.class_stat['TPR']
        self.history['class_F1_scores'] = confmat.class_stat['F1']
        self.history['class_supports'] = confmat.class_stat['P']
        
        
        
    def prune_weights(self, amount):
        parameters_to_prune = (
            (self.model.conv_1, 'weight'),
            (self.model.batch_norm_1, 'weight'),
            
            (self.model.res_block_1.conv_1, 'weight'),
            (self.model.res_block_1.batch_norm_1, 'weight'),
            (self.model.res_block_1.conv_2, 'weight'),
            (self.model.res_block_1.batch_norm_2, 'weight'),
            
            (self.model.res_block_2.conv_1, 'weight'),
            (self.model.res_block_2.batch_norm_1, 'weight'),
            (self.model.res_block_2.conv_2, 'weight'),
            (self.model.res_block_2.batch_norm_2, 'weight'),
            
            (self.model.conv_2, 'weight'),
            (self.model.batch_norm_2, 'weight'),

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
            torch.save(self.model.state_dict(), f'saved_models/init_param_models/CNN_1D_v{version_number}.pt')
        elif self.dataset == 'varying':
            torch.save(self.model.state_dict(), f'saved_models/zero_padded/CNN_1D_v{version_number}.pt')
        elif self.dataset == 'linear_interp':
            torch.save(self.model.state_dict(), f'saved_models/linear_interp/CNN_1D_v{version_number}.pt')
            
        print(f'CNN_1D_v{version_number} state_dict successfully saved')
        self.save_history(version_number, init)

    # =============================================================================
    # method to load model from state_dict
    # =============================================================================
    def load_model(self, version_number, init=False):
        self.version_number = version_number
        if init:
            self.model.load_state_dict(torch.load(f'saved_models/init_param_models/CNN_1D_v{version_number}.pt'))
        elif self.dataset == 'varying':
            self.model.load_state_dict(torch.load(f'saved_models/zero_padded/CNN_1D_v{version_number}.pt'))
        elif self.dataset == 'linear_interp':
            self.model.load_state_dict(torch.load(f'saved_models/linear_interp/CNN_1D_v{version_number}.pt'))
            
        print(f'CNN_1D_v{version_number} state dictionary successfully loaded')
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
            with open(f'saved_models/history/init_histories/CNN_1D_v{version_number}_history.pkl', 'wb') as f:
                pickle.dump(self.history, f)
        elif self.dataset == 'varying':
            with open(f'saved_models/history/zero_padded/CNN_1D_v{version_number}_history.pkl', 'wb') as f:
                pickle.dump(self.history, f)
        elif self.dataset == 'linear_interp':
            with open(f'saved_models/history/linear_interp/CNN_1D_v{version_number}_history.pkl', 'wb') as f:
                pickle.dump(self.history, f)
                
        print(f'CNN_1D_v{version_number} history saved')

    # =============================================================================
    # method that saves history to a pkl file
    # =============================================================================
    def load_history(self, version_number, init=False):
        if init:
            with open(f'saved_models/history/init_histories/CNN_1D_v{version_number}_history.pkl', 'rb') as f:
                self.history = pickle.load(f)
        elif self.dataset == 'varying':
            with open(f'saved_models/history/zero_padded/CNN_1D_v{version_number}_history.pkl', 'rb') as f:
                self.history = pickle.load(f)
        elif self.dataset == 'linear_interp':
            with open(f'saved_models/history/linear_interp/CNN_1D_v{version_number}_history.pkl', 'rb') as f:
                self.history = pickle.load(f)
                
        self.epochs = len(self.history['training_accuracy'])
        print(f'CNN_1D_v{version_number} history loaded')

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
                x = np.arange(len(y)) # number of epochsstrongest
                spline = make_interp_spline(x, y)
                x_ = np.linspace(x.min(), x.max(), 500)
                y_ = spline(x_)
                plt.plot(x_, y_)
                plt.title('Test loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
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

    def print_summary(self, print_cm=False, save_fig=False):
        self.confusion_matrix()
        print(f'\nModel: CNN_1D_v{self.version_number} -> Hyperparamters: \n'
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
            self.confusion_matrix(print_confmat=print_cm, save_fig=save_fig)


# =============================================================================
# instantiate K models and select the model with the highest validation accuracy
# =============================================================================
def nonrandom_init(K, dataset):
    print(f'Beginning random initalisation with {K} different models')
    records = {'index': None, 'highest_accuracy': 0}
    # models = []
    for k in range(K):
        model = CNN_1D_wrapper(CNN_1D, dataset='linear_interp', optimizer='AdamW', batch_size=128, combine=False)
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
    with open('saved_models/history/init_histories/CNN_1D_highest_idx.pkl', 'wb') as f:
        pickle.dump(records['index'], f)
        print(f"Highest_idx = {records['index']}, saved successfully")

    
# =============================================================================
# load the parameters of the model with the highest validation accuracy
# =============================================================================
def load_highest_model(model):
    with open('saved_models/history/init_histories/CNN_1D_highest_idx.pkl', 'rb') as f:
        highest_idx = pickle.load(f)
        print(f"Highest_idx = {highest_idx}, loaded successfully")
    model.load_model(highest_idx, init=True)
    

# =============================================================================
# driver code
# =============================================================================
if __name__ == "__main__":        
    # load the best randomly initialised network parameters for further training
    nonrand = False
    current_dataset = 'linear_interp'
    # current_dataset = 'varying'
    
    if nonrand:
        nonrandom_init(K=20, dataset=current_dataset)
        
    model = CNN_1D_wrapper(CNN_1D, dataset=current_dataset, optimizer='AdamW', batch_size=128, combine=False)
    # load_highest_model(model)
    
    # =============================================================================
    # testing zone
    # =============================================================================
    # model = CNN_1D_wrapper(CNN_1D, dataset='linear_interp', optimizer='AdamW', batch_size=128, combine=False)
    model.load_model(2)
    # model.fit(validate=True, epochs=1)
    # model.prune_weights(amount=0.21)
    model.predict()
    # model.print_params()
    model.print_summary(print_cm=True, save_fig=False)
    
    # model.print_params()
    # model.save_model(2)
    
    
    # model.plot('training_accuracy')
    # model.plot('validation_accuracy')
    # model.plot('training_loss')
    # model.plot('validation_loss')
    
    # model.plot('accuracy')
    # model.plot('loss')
