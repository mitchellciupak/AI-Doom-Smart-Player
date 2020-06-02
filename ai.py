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
    
    def forward(self x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) 
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) 
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) 
        x = x.view(x.size(0), -1) # Flatten multi channel convolution
        x = F.relu(slef.fc1(x))
        x = self.fc2(x)
        return x
    
# Create Player 

class SoftMaxPlayer(nn.Module):
    
    def _init_(self, T):
        super(SoftMaxPlayer, self).__init__()
        self.T = T
    
    def forward(self outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions
    
# Create AI

class AI:
    def _init_(self, brain, body):
        self.brain = brain
        self.body = body
        
    def __call__(self, inputs):
        # Convert Images into Torch Structure
        input = Variable(tourch.from_numpy(np.array(inputs, dtype = np.float32)))
        
        # Propegate signals
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()
    
# Train AI (Deep Convolutional Q-Learning)
        
## Setup Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n

##Init Auto Player Objects
cnn = CNN(number_actions)
softMaxPlayer= SoftMaxPlayer(T = 1.0) #Small Temp
ai = AI(brain = cnn, body = softMaxPlayer)

##Init Expericne Replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10) #Learn every 10 transitions
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)
    
    
        