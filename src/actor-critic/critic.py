# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg
# Description: critic

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  torch.autograd import Variable

class CNet(nn.Module):

    def __init__(self, state_space):
        super(CNet, self).__init__()
        self.state_space = state_space
        self.fc1 = nn.Linear(self.state_space, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        fc1 = self.fc1(x)
        relu1 = F.relu(fc1)
        fc2 = self.fc2(relu1)
        return fc2

class Critic(object):

    def __init__(self, Net, lr, gamma):
        self.Net = Net
        self.lr = lr
        self.gamma = gamma
        self.optim = torch.optim.Adam(self.Net.parameters(), lr = self.lr)

    def learn(self, current_state, reward, next_state):
        current_state = torch.from_numpy(current_state)
        current_state = current_state.unsqueeze(0)
        current_state = Variable(current_state.float().cuda())

        next_state = torch.from_numpy(next_state)
        next_state = next_state.unsqueeze(0)
        next_state = Variable(next_state.float().cuda())

        current_value = self.Net(current_state)
        next_value = self.Net(next_state)

        td_error = reward + self.gamma * next_value - current_value
        loss = torch.mean(torch.mul(td_error, td_error))

        self.optim.zero_grad()
        loss.backward(retain_graph = True)
        self.optim.step()

        return td_error

