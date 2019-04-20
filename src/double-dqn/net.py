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

        self.fc1 = nn.Linear(self.state_space, 24) # 10 hidden units
        self.fc2 = nn.Linear(24, 24) # 10 hidden units
        self.fc3 = nn.Linear(24, self.action_space)
        

    def forward(self, x):
        fc1 = self.fc1(x)
        relu1 = F.relu(fc1)
        fc2 = self.fc2(relu1)
        relu2 = F.relu(fc2)
        output = self.fc3(relu2)

        return output


