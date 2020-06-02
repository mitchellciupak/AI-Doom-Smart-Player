# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:45:22 2020

@author: mjcre
"""

# Import
## Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

## Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

## Importing the other Python files
import experience_replay, image_preprocessing

# Create CNN
## Class Init
class CNN(nn.Module):
    
    def _init_(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) # 1 Black and White image, 32 detected features, 5x5 feature detector 
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) # In channels are now out channels of last convolution connection 
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        self.fc1 = nn.Linear(in_features = self.count_neurons((1,, 80, 80)), out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
        
    def count_neurons(self, image_dim):
        # image_dim = tuple of image demensions
        
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) # Max Pooling with 3 Kernel Size, 2 Strides
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) # Max Pooling with 3 Kernel Size, 2 Strides
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) # Max Pooling with 3 Kernel Size, 2 Strides
        return x.data.view(1, -1).size(1) #Size of neuron array