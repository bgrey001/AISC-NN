#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:00:35 2022

@author: BenedictGrey

Official PyTorch demo for LeNet and Net

LeNet is one of the early CNNs for MNIST

"""
import torch
import torch.nn as nn # parent object for pytorch models
import torch.nn.functional as F # activation function

class LeNet(nn.Module):
    
    def __init__(self): # instantiate layers
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3) # C1
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3) # C3
        self.fc1 = nn.Linear(in_features=16 * 6 * 6, out_features=120)  # 6*6 from image dimension F5
        self.fc2 = nn.Linear(in_features=120, out_features=84) # F6
        self.fc3 = nn.Linear(in_features=84, out_features=10) # OUTPUT
        
    def forward(self, x): # This is where the computation happens, an input is passed through the network layers and various functions to generate an output
        # Max pooling over a (2, 2) window    
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
        
        
# =============================================================================
# net = LeNet()
# print(net) # what does the object tell us about itself?
# 
# input = torch.rand(1, 1, 32, 32) # stand in for a 32x32 black and white image
# print('\nImage batch shape: ' + str(input.shape))
# 
# output = net(input) # we don't call forward() directly
# print('\nRaw output: ' + str(output) + '\nshape: ' + str(output.shape))        
# =============================================================================
        
"""
PyTorch models assume they are working on batches of data, this is the first value in the tensor

Ask the model for an inference by calling it like a function: net(input). 
The output of this call represents the model's confidence that the input represents a particular digit. 
Obviously, nothing has been learned yet. 
Looking at the shape of output, we can see it also has a batch dimension, the size of which should always match the input batch dimension

"""
        

# =============================================================================
# Datasets and Dataloaders
# =============================================================================
        

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), #c onverts images loaded by Pillow into PyTorch tensors.
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]) # adjusts the values of the tensor so that their average is zero and their standard deviation is 0.5. 


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5 # unnormalise
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    
# get some random images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))























        