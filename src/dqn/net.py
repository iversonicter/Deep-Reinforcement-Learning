import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class Net(nn.Module):

    def __init__(self, state_space, action_space):
        super(Net, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.state_space, 10) # 10 hidden units
        self.fc2 = nn.Linear(10, self.action)
        

    def forward(self, x):
        x = nn.ReLU(self.fc1(x))
        output = self.fc2(x)

        return output


