import os
import logging

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class net2layer(nn.Module):
    def __init__(self):
        super(net2layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # print (np.shape (x))
        x = x.view(-1, 9216)
        # x = torch.flatten(x, 1)
        fea = self.fc1(x)
        # x = F.relu(fea)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        
        return fea