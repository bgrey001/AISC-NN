#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:56:18 2022

@author: benedict
"""


# =============================================================================
# model class inherits from torch Module class
# =============================================================================
class CNN_1D_v1(nn.Module):

    # =============================================================================
    # class attributes
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================================================================
    # constructor
    # =============================================================================
    def __init__(self, n_features, n_classes, seq_length, conv_l1, kernel_size, pool_size):
        super(CNN_1D_v1, self).__init__()

        # calculate channel sizes for the different convolution layers
        conv_l2 = 2 * conv_l1
        conv_l3 = 2 * conv_l2
        conv_l4 = 2 * conv_l3

        # conv layers 1
        self.batch_norm_1 = nn.BatchNorm1d(n_features)
        self.conv_1 = nn.Conv1d(in_channels=n_features, out_channels=conv_l1, kernel_size=kernel_size)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool1d(pool_size)    

        # conv layers 2
        self.batch_norm_2 = nn.BatchNorm1d(conv_l1)
        self.conv_2 = nn.utils.weight_norm(nn.Conv1d(in_channels=conv_l1, out_channels=conv_l2, kernel_size=kernel_size))
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool1d(pool_size)


        # conv layers 3
        self.batch_norm_3 = nn.BatchNorm1d(conv_l2)
        self.conv_3 = nn.utils.weight_norm(nn.Conv1d(in_channels=conv_l2, out_channels=conv_l3, kernel_size=kernel_size))
        self.relu_3 = nn.ReLU()
        self.pool_3 = nn.MaxPool1d(pool_size)
        
        # conv layers 4
        self.batch_norm_4 = nn.BatchNorm1d(conv_l3)
        self.conv_4 = nn.utils.weight_norm(nn.Conv1d(in_channels=conv_l3, out_channels=conv_l4, kernel_size=kernel_size))
        self.relu_4 = nn.ReLU()
        self.pool_4 = nn.AvgPool1d(pool_size)

        # configure transformed dimensions of the input as it reaches the fully connected layer
        conv_l1_dim = math.floor((seq_length - (kernel_size - 1))/ pool_size)
        conv_l2_dim = math.floor((conv_l1_dim - (kernel_size - 1)) / pool_size)
        conv_l3_dim = math.floor((conv_l2_dim - (kernel_size - 1)) / pool_size)
        conv_l4_dim = math.floor((conv_l3_dim - (kernel_size - 1)) / pool_size)
        
        # flat_size = conv_l1 * conv_l1_dim
        flat_size = conv_l4 * conv_l4_dim

        # flatten and prediction layers
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(flat_size, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    # =============================================================================
    # forward propagation method
    # =============================================================================
    def forward(self, input_x):

        # residual = input_x
        # conv layers 1
        input_x = self.batch_norm_1(input_x)
        input_x = self.conv_1(input_x)
        input_x = self.relu_1(input_x)
        input_x = self.pool_1(input_x)

        # conv layers 2
        input_x = self.batch_norm_2(input_x)
        input_x = self.conv_2(input_x)
        input_x = self.relu_2(input_x)
        input_x = self.pool_2(input_x)

        # # conv layers 3
        input_x = self.batch_norm_3(input_x)
        input_x = self.conv_3(input_x)
        input_x = self.relu_3(input_x)
        input_x = self.pool_3(input_x)
        
        # # conv layers 4
        input_x = self.batch_norm_4(input_x)
        input_x = self.conv_4(input_x)
        input_x = self.relu_4(input_x)
        input_x = self.pool_4(input_x)

        # flatten and prediction layers
        input_x = self.flatten(input_x)
        input_x = F.relu(self.fc_1(input_x))
        input_x = F.relu(self.fc_2(input_x))
        input_x = F.relu(self.fc_3(input_x))
        output = self.softmax(input_x)
        
        # output += residual

        return output