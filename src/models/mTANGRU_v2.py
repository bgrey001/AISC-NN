#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:57:22 2023
@author: Benedict Grey

DISCLAIMER:
Script for the implementation of Shukla and Marlin's mTAN encoder classification network
All credit goes to the original authors, this is merely an modification of the original code
https://github.com/reml-lab/mTAN

"""

# =============================================================================
# dependencies
# =============================================================================
from pycm import ConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.interpolate import make_interp_spline
from datetime import datetime
import math

import AIS_loader as data_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune

sns.set_style("darkgrid") 

# =============================================================================
# mTAN module
# =============================================================================
class multiTimeAttention(nn.Module):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    def __init__(self, input_dim, nhidden, embed_time, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        torch.cuda.empty_cache()
        dim = value.size(-1)
        d_k = query.size(-1)
        # scores = keys * queries
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        # probabilities, aka the scores softmaxed
        p_attn = F.softmax(scores, dim = -2).to(self.device)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # probabilities * values, this is the sum of the weighted averages
        return torch.sum(p_attn*value.unsqueeze(-3).to(self.device), -2), p_attn.to(self.device)
    
    
    """
    mTAN(t, s)
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        # 
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


# =============================================================================
# full mTAN classification network
# =============================================================================
class mTAN_enc(nn.Module):
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
 
    # =============================================================================
    # constructor
    # =============================================================================
    def __init__(self, input_dim, query, n_gru_layers, nhidden, embed_time, num_heads=1, learn_emb=True, freq=10., bidirectional=True, n_classes=6):
        super(mTAN_enc, self).__init__()
        
        bi_dim = 1
        if bidirectional: bi_dim = 2
        self.bi_dim = bi_dim
        self.n_classes = n_classes
        self.query = query
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.nhidden = nhidden
        self.att = multiTimeAttention(input_dim, nhidden, embed_time, num_heads)
        self.GRU = nn.GRU(nhidden, nhidden, bidirectional=bidirectional, num_layers=n_gru_layers, batch_first=True)
        
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
            
        self.classifier = nn.Sequential(
            nn.Linear(nhidden*self.bi_dim, 128), 
            nn.Linear(128, 64),
            nn.Linear(64, self.n_classes))
            
        

    # =============================================================================
    # method that learns the time embeddings as weights
    # =============================================================================
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
    # =============================================================================
    # method that generates positional encodings to provide sequential semantics for the model
    # =============================================================================
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
       
    def forward(self, x, time_steps):
        print(x.shape)
        time_steps = time_steps.to(self.device)
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device) 
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        att_out = self.att(query=query, key=key, value=x, mask=None) # key and value are both the embeddings and the value is the feature vector
        gru_out, hidden = self.GRU(att_out)
        class_in = torch.cat((gru_out[:, -1, :self.nhidden], gru_out[:, 0, self.nhidden:]), dim=1)        
        out = self.classifier(class_in)
        return out
    
    
# =============================================================================
# wrapper class for an instance of the mTAN_RNN model
# =============================================================================
class mTAN_wrapper():
    # =============================================================================
    # Class attributes
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # =============================================================================
    # Data attributes
    # =============================================================================
    data_ver = '1'
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
    def __init__(self, mTAN, version_number, query, dataset, n_gru_units, hidden_dim, embed_time, optimizer, bidirectional, batch_size, combine=False):
        
        # init class members
        self.dataset = dataset
        self.version_number = version_number
        self.batch_size = batch_size
        self.query = query
        self.min_val_loss = np.inf
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
        if combine: self.train_data = data_module.AIS_loader(choice=self.dataset, split='train', version=self.data_ver, split_2='valid')
        else:
            self.train_data = data_module.AIS_loader(choice=self.dataset, split='train', version=self.data_ver)
            self.valid_data = data_module.AIS_loader(choice=self.dataset, split='valid', version=self.data_ver)
        self.test_data = data_module.AIS_loader(choice=self.dataset, split='test', version=self.data_ver)
        
        self.n_features = 4
        self.n_classes = self.train_data.n_classes
        self.seq_length = self.train_data.seq_length

        self.model = mTAN_enc(input_dim=self.n_features, 
                         query=self.query,
                         nhidden=hidden_dim, 
                         n_gru_layers=n_gru_units, 
                         n_classes=self.n_classes, 
                         bidirectional=bidirectional,
                         embed_time=embed_time, 
                         num_heads=1, 
                         learn_emb=True, 
                         freq=10.).to(self.device)

        match optimizer:
            case 'AdamW': self.optimizer = optim.AdamW(self.model.parameters(), lr=self.eta, weight_decay=self.alpha)
            case 'SGD': self.optimizer = optim.SGD(self.model.parameters(), lr=self.eta, weight_decay=self.weight_decay, momentum=self.alpha)
            case 'RMSprop': self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.eta, weight_decay=self.weight_decay, momentum=self.alpha)

        self.optim_name = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def class_from_output(self, output, is_tensor):
        if is_tensor: class_index = torch.argmax(output).item()
        else: class_index = int(output)
        return self.data_loader.return_classes()[class_index]

    # =============================================================================
    # Train model
    # =============================================================================
    def fit(self, validate, epochs=None):
        if epochs == None: epochs = self.epochs
        else: self.epochs = epochs
        avg_val_loss = np.inf
        train_print_steps = 600
        val_print_steps = plot_steps = 400
        train_loss = valid_loss = 0

        # instantiate DataLoader
        train_generator = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.train_data.GRU_collate, drop_last=True)
        if validate: valid_generator = DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.valid_data.GRU_collate, drop_last=True)

        for epoch in range(epochs):
            start_time = datetime.now()   
            aggregate_correct = 0
            v_correct = 0
            train_loss_counter = 0
            valid_loss_counter = 0
            # =============================================================================
            # training loop
            # =============================================================================
            for index, (seqs, time_steps, labels, lengths) in enumerate(train_generator):
                seqs, time_steps, labels = seqs.to(self.device), time_steps.to(self.device), labels.to(self.device)
                self.model.train()
                out = self.model(seqs, time_steps)
                loss = self.criterion(out, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                outputs = torch.argmax(out, dim=1)
                aggregate_correct += (((outputs == labels).sum().item()) / len(labels)) * 100
            
                if index == 0 and epoch == 0:
                    first_accuracy = aggregate_correct
                    print(f'Initial accuracy = {first_accuracy}')
                    
                if (index + 1) % plot_steps == 0:
                    if (index + 1) % train_print_steps == 0:
                        print(f'Epoch {epoch}, batch number: {index + 1}, training loss = {loss}')
                train_loss_counter += 1
                train_loss += loss.item()
                            
            train_accuracy = aggregate_correct / (len(train_generator))
            self.training_accuracies.append(train_accuracy)
            self.training_losses.append(train_loss/train_loss_counter)
            self.history['training_accuracy'].append(train_accuracy)
            self.history['training_loss'].append(train_loss/train_loss_counter)
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
                for valid_seqs, valid_time_steps, valid_labels, lengths in valid_generator:
                    
                    self.model.eval()
                    valid_seqs, valid_labels = valid_seqs.to(self.device), valid_labels.to(self.device)
                    with torch.no_grad():
                        # forward propagataion
                        valid_output = self.model(valid_seqs, valid_time_steps)
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
                avg_val_loss = valid_loss / valid_loss_counter
                self.validation_accuracies.append(val_accuracy)
                self.validation_losses.append(avg_val_loss)
                self.history['validation_accuracy'].append(val_accuracy)
                self.history['validation_loss'].append(avg_val_loss)
                print("========================================================================================================================================================== \n" +
                      f" ------------------> validation accuracy = {val_accuracy}, average validation loss = {valid_loss / valid_loss_counter} <------------------" +
                      "========================================================================================================================================================== \n")

                # model checkpoints or callbacks
                if avg_val_loss < self.min_val_loss:
                    self.min_val_loss = avg_val_loss
                    print(f'CHECKPOINT: new minimum val loss {self.min_val_loss}, checkpoint created.\n')
                    self.checkpoint()
                                    
                valid_loss = 0
                valid_loss_counter = 0
            else: self.predict()
            self.confusion_matrix(valid=validate)
            print(f'Class F1-scores: {self.history["class_F1_scores"]}\n')
            end_time = datetime.now()
            print(f'Epoch duration: {(end_time - start_time)}\n')



    # =============================================================================
    # method that tests model on test data
    # =============================================================================
    def predict(self):
        test_correct = test_loss_counter = test_loss = 0
        test_print_steps = 400
        test_generator = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.test_data.GRU_collate, drop_last=True)
        test_index = 0
        for test_seqs, test_time_steps, test_labels, lengths in test_generator:
            self.model.eval()
            with torch.no_grad():
                test_seqs, test_time_steps, test_labels = test_seqs.to(self.device), test_time_steps.to(self.device), test_labels.to(self.device)
                test_output = self.model(test_seqs, test_time_steps)
                t_loss = self.criterion(test_output, test_labels)
                test_outputs = torch.argmax(test_output, dim=1)
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
        if valid: test_generator = DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.test_data.GRU_collate, drop_last=True)
        else: test_generator = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.test_data.GRU_collate, drop_last=True)

        for test_seqs, test_time_steps, test_labels, lengths in test_generator:

            self.model.eval()
            with torch.no_grad():
                test_seqs, test_time_steps, test_labels = test_seqs.to(self.device), test_time_steps.to(self.device), test_labels.to(self.device)
                test_output = self.model(test_seqs, test_time_steps)
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
            plt.savefig(f'mTAN_v{self.version_number}.png', dpi=300)
            plt.savefig(f'../../plots/mTAN/v{self.version_number}/confmat_mTAN_v{self.version_number}.png', dpi=300)
        
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
    # method to save the model and its history if it's produced the best results so far
    # =============================================================================
    def checkpoint(self):
        self.save_model(version_number=self.version_number, condition='checkpoint')
        
        
    # =============================================================================
    # method to save model to state_dict
    # =============================================================================
    def save_model(self, version_number, condition):
        self.version_number = version_number
        match condition:
            case 'init':
                torch.save(self.model.state_dict(), f'saved_models/init_param_models/mTANGRU_v{version_number}.pt')
                print(f'mTANGRU_v{version_number} state_dict successfully saved')
            case 'checkpoint':
                torch.save(self.model.state_dict(), f'saved_models/checkpoints/mTANGRU_cp_v{version_number}.pt')
                
            case 'final_model':
                torch.save(self.model.state_dict(), f'saved_models/mTANGRU_v{version_number}.pt')
                print(f'mTANGRU_v{version_number} state_dict successfully saved')
            
        self.save_history(version_number, condition)

    # =============================================================================
    # method to load model from state_dict
    # =============================================================================
    def load_model(self, version_number, condition):
        self.version_number = version_number
        match condition:
            case 'init':
                self.model.load_state_dict(torch.load(f'saved_models/init_param_models/mTANGRU_v{version_number}.pt'))
                print(f'mTANGRU_v{version_number} state_dict successfully loaded')
            case 'checkpoint':
                self.model.load_state_dict(torch.load(f'saved_models/checkpoints/mTANGRU_cp_v{version_number}.pt'))
                print(f'Checkpoint mTANGRU_cp{version_number} state_dict successfully saved')
            case 'final_model':
                self.model.load_state_dict(torch.load(f'saved_models/mTANGRU_v{version_number}.pt'))
                print(f'mTANGRU_v{version_number} state_dict successfully saved')
        
        self.load_history(version_number, condition)


    # =============================================================================
    # method that saves history to a pkl file
    # =============================================================================
    def save_history(self, version_number, condition):
        match condition:
            case 'init':
                with open(f'saved_models/history/init_histories/mTANGRU_v{version_number}_history.pkl', 'wb') as f:
                    pickle.dump(self.history, f)
                print(f'mTANGRU_v{version_number} history saved')
            case 'checkpoint':
                with open(f'saved_models/history/init_histories/checkpoints/mTANGRU_cp_v{version_number}_history.pkl', 'wb') as f:
                    pickle.dump(self.history, f)
            case 'final_model':
                with open(f'saved_models/history/mTANGRU_v{version_number}_history.pkl', 'wb') as f:
                    pickle.dump(self.history, f)
                print(f'mTANGRU_v{version_number} history saved')
                
            
    # =============================================================================
    # method that saves history to a pkl file
    # =============================================================================
    def load_history(self, version_number, condition):
        match condition:
            case 'init':
                with open(f'saved_models/history/init_histories/mTANGRU_v{version_number}_history.pkl', 'rb') as f:
                    self.history = pickle.load(f)
                print(f'mTANGRU_v{version_number} history loaded')
            case 'checkpoint':
                with open(f'saved_models/history/init_histories/checkpoints/mTANGRU_cp_v{version_number}_history.pkl', 'rb') as f:
                    self.history = pickle.load(f)
                print(f'Checkpoint mTANGRU_cp{version_number} history loaded')
            case 'final_model':
                with open(f'saved_models/history/mTANGRU_v{version_number}_history.pkl', 'rb') as f:
                    self.history = pickle.load(f)
                print(f'mTANGRU_v{version_number} history loaded')
            

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
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.legend()
                plt.show()
            
                
    # =============================================================================
    # plot given metric
    # =============================================================================

    def print_summary(self, print_cm=False):
        self.confusion_matrix()
        print(f'\nModel: mTAN_v{self.version_number} -> Hyperparamters: \n'
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
        if print_cm: self.confusion_matrix(print_confmat=(True))


# =============================================================================
# instantiate K models and select the model with the highest validation accuracy
# =============================================================================
def nonrandom_init(K, dataset):
    print(f'Beginning random initalisation with {K} different models')
    records = {'index': None, 'lowest_loss': np.inf}
    ref_time_pts = 1440 # number of minutes in a day
    for k in range(K):
        model = mTAN_wrapper(mTAN_enc,
                            version_number=99,
                            query=torch.linspace(0, 1., ref_time_pts),
                            dataset=dataset, 
                            n_gru_units=2, 
                            embed_time=256,
                            hidden_dim=64,
                            optimizer='AdamW', 
                            bidirectional=True, 
                            batch_size=10, 
                            combine=False)
        print(f'MODEL {k + 1} -------------------------------->')
        model.fit(validate=True, epochs=2)
        if min(model.history['validation_loss']) < records['lowest_loss']:
            records['index'] = k + 1
            print(f'New highest record: model {k + 1}')
            records['lowest_loss'] = min(model.history['validation_loss'])
        model.save_model(k + 1, condition='init')
        del model
        
    # save highest index 
    with open('saved_models/history/init_histories/mTAN_highest_idx.pkl', 'wb') as f:
        pickle.dump(records['index'], f)
        print(f"Highest_idx = {records['index']}, saved successfully")
    

    
# =============================================================================
# load the parameters of the model with the highest validation accuracy
# =============================================================================
def load_highest_model(model):
    with open('saved_models/history/init_histories/mTAN_highest_idx.pkl', 'rb') as f:
        highest_idx = pickle.load(f)
        print(f"Highest_idx = {highest_idx}, loaded successfully")
    model.load_model(highest_idx, condition='init')
    
    
# =============================================================================
# driver code
# =============================================================================
if __name__ == "__main__":
    nonrand = False
    current_dataset = 'non_linear'
    
    if nonrand: nonrandom_init(K=10, dataset=current_dataset)
    
    vn = 99
    ref_time_pts = 1440 # number of minutes in a day
    model = mTAN_wrapper(mTAN_enc, 
                        version_number=vn,
                        query=torch.linspace(0, 1, ref_time_pts),
                        dataset=current_dataset, 
                        n_gru_units=2, 
                        embed_time=256,
                        hidden_dim=64,
                        optimizer='AdamW', 
                        bidirectional=True, 
                        batch_size=10, 
                        combine=False)
    
    # load_highest_model(model)
    
    # =============================================================================
    # testing zone
    # =============================================================================
    # model.load_model(version_number=3, condition='final_model')
    # model.load_model(version_number=7, condition='init')
    model.load_model(version_number=5, condition='checkpoint')
    model.version_number = 6
    h = model.history
    model.history['validation_loss'] = list(dict.fromkeys(model.history['validation_loss']))
    model.history['validation_accuracy'] = list(dict.fromkeys(model.history['validation_accuracy']))
    model.history['training_loss'] = list(dict.fromkeys(model.history['training_loss']))
    model.history['training_accuracy'] = list(dict.fromkeys(model.history['training_accuracy']))
    # list(dict.fromkeys('abracadabra'))
    # min(model.history['validation_loss'])
    # model.min_val_loss
    # model.fit(validate=True, epochs=60)
# =============================================================================
#     # model.prune_weights(amount=0.2)
#     model.predict()
#     model.print_summary(print_cm=True)
#     # model.confusion_matrix()
#     model.save_model(version_number=6, condition='final_model')
#     
#     
#     model.prune_weights(amount=0.2)
# 
#     
#     
#     # for i in range(1, 11):
#     #     model.load_model(version_number=i, condition='init')
#     #     print(model.history["validation_accuracy"])
#     #     print(model.history["class_F1_scores"], '\n')
#         
#     
#     
#     # model.plot('training_accuracy')
#     # model.plot('validation_accuracy')
#     # model.plot('training_loss')
#     # model.plot('validation_loss')
#     model.plot('accuracy')
#     model.plot('loss')      
# 
# 
# =============================================================================




