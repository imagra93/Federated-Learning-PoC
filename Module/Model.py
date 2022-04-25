# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:43:27 2022

@author: TEYYC-TL6
"""


from torch import nn
import torch.nn.functional as F


class CNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5,5)) # InputChannels / OutputChannels / kernel_size / Stride = 1 (default) | Padding = 0 (default)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d() # Randomly zero out entire channels
        self.fc1 = nn.Linear(320, 50) # Input neurons | Output Neurons.
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        #
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        #
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) # Reshape to 1 dimensión. -1 is default.
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, inplace = False) # Standard Dropout. Only at training.
        #
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1) # Applies a softmax followed by a logarithm
        #
        return x