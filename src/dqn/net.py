# Author : Wang Yongjie
# Email  : yongjie.wang@ntu.edu.sg
# Description: DQN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):

    def __init__(self, state_space, action_space):
        super(Net, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.state_space, 10) # 10 hidden units
        self.fc2 = nn.Linear(10, self.action_space)
        

    def forward(self, x):
        fc1 = self.fc1(x)
        relu1 = F.relu(fc1)
        output = self.fc2(fc1)

        return output


