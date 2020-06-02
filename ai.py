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
        self.convolution1
        self.convolution2
        self.convolution3
        self.fc1
        self.fc2