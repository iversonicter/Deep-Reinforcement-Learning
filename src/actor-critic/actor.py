# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg
# Description: actor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  torch.autograd import Variable


class ANet(nn.Module):
    def __init__(self, state_space, action_space):
        super(ANet, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.state_space, 20)
        self.fc2 = nn.Linear(20, self.action_space)

    def forward(self, x):
        fc1 = self.fc1(x)
        relu1 = F.relu(fc1)
        fc2 = self.fc2(relu1)
        
        return fc2

class Actor(object):

    def __init__(self, Net, lr, exploration_rate = 0.999):
        self.Net = Net
        self.lr = lr
        self.optim = torch.optim.Adam(self.Net.parameters(), lr = self.lr)
        self.exploration_rate = exploration_rate

    def learn(self, state, action, td_error):
        ## compute the gradient and update the model
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        state = Variable(state.float().cuda())
        prediction = self.Net(state)
        prediction = F.softmax(prediction)
        log_prob = prediction[0][action] #
        loss = torch.mean(log_prob * td_error)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def choose_action(self, state):
        ## select an action from current state
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        state = Variable(state.float().cuda())
        prediction = self.Net(state)
        prediction = F.softmax(prediction)
        prediction = prediction[0].cpu().detach().numpy()
        if np.random.rand() > self.exploration_rate:
            action = np.random.choice(range(prediction.shape[0]), p = prediction)
        else:
            action = np.argmax(prediction)

        self.exploration_rate = self.exploration_rate * 0.99
        return action

