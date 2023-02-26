#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:21:50 2023
@author: benedict

DISCLAIMER:
Script for the implementation of Shukla and Marlin's mTAN encoder classification network (without the decoder part)
All credit goes to the original authors, this is merely an modification of the original code


"""

# =============================================================================
# dependencies
# =============================================================================
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time
import numpy as np
import pandas as pd
import math
from random import SystemRandom


import AIS_loader as loader

# import models
# import utils


# =============================================================================
# mTAN module
# =============================================================================
class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2).to(self.device)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3).to(self.device), -2), p_attn.to(self.device)
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


# =============================================================================
# full mTAN classification network
# =============================================================================
class enc_mtan(nn.Module):
 
    # =============================================================================
    # constructor
    # =============================================================================
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
        super(enc_mtan, self).__init__()
        # assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        # self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.att = multiTimeAttention(input_dim, nhidden, embed_time, num_heads)
        self.gru = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
            
        self.classifier = nn.Sequential(
            nn.Linear(nhidden*2, 128), 
            nn.Linear(128, 64),
            nn.Linear(64, 6))
            

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
    # method that uses positional encodings to provide sequential semantics for the model
    # =============================================================================
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
       
    def forward(self, x, time_steps):
        batch = x.size(0)
        # time_steps = time_steps.cpu()
        time_steps = time_steps
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device) 
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
        att_out = self.att(query=key, key=key, value=x, mask=None)
        gru_out, hidden = self.gru(att_out)
        hidden = hidden.view(1, 2, 30, self.nhidden)
        hidden = hidden[-1]
        hidden_forward, hidden_backward = hidden[0], hidden[1]
        class_in = torch.cat((hidden_forward, hidden_backward), dim = 1)
        out = self.classifier(class_in)
        return out
    
    
if __name__ == '__main__':
    # torch.cuda.memory_summary(device=None, abbreviated=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 30
    seq_length = 2931
    n_features = 4
    
    dataset = loader.AIS_loader(choice='non_linear', split='train', version=1)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=(dataset.GRU_collate), drop_last=True)
    # dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, collate_fn=(dataset.CNN_collate))
    # dataset.visualise_tensor(250)
    
    # declare encoder
    model = enc_mtan(input_dim=n_features, nhidden=64, embed_time=16, num_heads=1, learn_emb=True, freq=10.).to(device)
    eta = 3e-4
    alpha = 1e-4
    
    classify_pertp = False
    epochs = 25
    
    params = (list(model.parameters()))
    # print('parameters:', utils.count_parameters(rec))
    optimizer = optim.AdamW(params, lr=eta, weight_decay=alpha)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    total_time = 0.
    
    training_accuracies = []
    training_losses = []

    
    for epoch in range(epochs):
        
        aggregate_correct = 0
        v_correct = 0
        train_loss_counter = 0
        valid_loss_counter = 0
        start_time = time.time()
        
        train_print_steps = 100
        plot_steps = 100
        train_loss = valid_loss = 0
        
        index = 0
        for batch_seqs, batch_time_steps, labels, lengths in dataloader:
            
            batch_seqs, batch_time_steps, labels = batch_seqs.to(device), batch_time_steps.to(device), labels.to(device)
            # feature = batch_seqs[0]
            # print(batch_time_steps.shape)
            # time_steps = batch_time_steps[0]
            # batch_mask = torch.ones(batch_size, seq_length, n_features).to(device)
            # out = model(torch.cat((batch_seqs, batch_mask), 2), batch_time_steps)
            out = model(batch_seqs, batch_time_steps)
            
            loss = criterion(out, labels)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # train_loss += loss.item() * batch_size
            # train_acc += torch.mean((out.argmax(1) == label).float()).item() * batch_size
            # train_n += batch_size
            
            outputs = torch.argmax(out, dim=1)
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
            
        train_accuracy = aggregate_correct / (len(dataloader))
        training_accuracies.append(train_accuracy)
        training_losses.append(train_loss/train_loss_counter)
        print("========================================================================================================================================================== \n" +
              f"------------------> training accuracy = {train_accuracy}, average training loss = {train_loss / train_loss_counter} <------------------\n" +
              "========================================================================================================================================================== \n")

        train_loss = 0
        train_loss_counter = 0
            
        # total_time += time.time() - start_time
        # val_loss, val_acc, val_auc = utils.evaluate_classifier(model, val_loader, args=args, dim=dim)
        # best_val_loss = min(best_val_loss, val_loss)
        # test_loss, test_acc, test_auc = utils.evaluate_classifier(rec, test_loader, args=args, dim=dim)
        # print('Iter: {}, loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
        #       .format(itr, train_loss/train_n, train_acc/train_n, val_loss, val_acc, test_acc, test_auc))
        
    
    
    
    

