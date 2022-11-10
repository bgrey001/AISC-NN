#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 09:27:28 2022

@author: benedict
"""
import torch
from torch import nn
from torch.utils.data import DataLoader # primitive DataLoader for working with data, this one wraps an iterable around the Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


import numpy as np

# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root = 'data',
    train=True,
    download=True,
    transform=ToTensor(),
    )

test_data = datasets.FashionMNIST(
    root = 'data',
    train=False,
    download=True,
    transform=ToTensor(),
    )


batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)



for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


a = X.detach().numpy()
b = a[]




# =============================================================================
# creating models
# =============================================================================

# get cpu or gpu for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')



