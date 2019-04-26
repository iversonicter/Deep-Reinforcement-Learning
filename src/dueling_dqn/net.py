# Author : Wang Yongjie
# Email  : yongjie.wang@ntu.edu.sg
# Description: dueling DQN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):

    def __init__(self, state_space, action_space):
        super(Net, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.state_space, 24) # 10 hidden units
        self.fc2 = nn.Linear(24, self.action_space) # value
        self.fc3 = nn.Linear(24, self.action_space) # advantage
        

    def forward(self, x):
        fc1 = self.fc1(x)
        relu1 = F.relu(fc1)
        value = self.fc2(relu1)
        advantage = self.fc3(relu1)
        output = value + (advantage - torch.mean(advantage, dim = -1))

        return output
