#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:53:17 2022

@author: BenedictGrey

Are CNNs better at dealing with class imbalance?

Relevant information is distilled - magnified and refined - whilst irrelevant information is filtered out

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class 1DCNN(nn.Module):
    # =============================================================================
    # class attributes
    # =============================================================================
    
    def __init__(self):
        