#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 09:19:33 2022

@author: benedict

PyTorch demo for a CNN with training

"""
#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# data processing
# =============================================================================

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # download the CIFAR10 dataset from torchvision and apply the transformation from lines above
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#visualise the data
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # add layers to the model (instance of Net class (inherited from nn.Module))
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # view() reshapes the tensor like numpy's reshape() but without copying memory
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        return x
        
        
net = Net() # create an instance of this model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)

summary(model, (3, 32, 32))

# =============================================================================
# for p in net.parameters():
#     print(p.size())
# 
# =============================================================================

# =============================================================================
# loss function and optimiser or learning algorithm
# =============================================================================
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# =============================================================================
# 
# # =============================================================================
# # training
# # =============================================================================
# 
# 
# 
# for epoch in range(2):
#     
#     running_loss = 0.0
#     
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs
#         inputs, labels = data
#          
        # zero the parameter gradients
#        optimizer.zero_grad()

        # forward + backward + optimize
#        outputs = net(inputs)
#        loss = criterion(outputs, labels)
 #       loss.backward()
 #       optimizer.step()
# 
#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#         
# print('Finished training')
# 
# 
# # =============================================================================
# # testing
# # =============================================================================
# 
# 
# correct = 0
# total = 0
# 
# with torch.no_grad():
#     for data in testloader:
# 
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         
# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
# 
# 
# =============================================================================



















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        