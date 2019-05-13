# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg
# Description: actor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, state_space, action_space):
        super(Net, self).__init__()
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

    def __init__(self, Net, lr):
        self.Net = Net
        self.lr = lr
        self.optim = torch.optim.Adam(self.Net.parameters(), lr = self.lr)

    def learn(self, state, action, td_error):
        ## compute the gradient and update the model
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        state = Variable(state.float().cuda())
        prediction = self.Net(state)
        prediction = F.softmax(prediction)
        log_prob = prediction[action] #
        loss = log_prob * td_error
        self.optim.zero_grad()
        loss .backward()
        self.optim.step()

    def choose_action(self, state):
        ## select an action from current state
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        state = Variable(state.float().cuda())
        prediction = self.Net(state)
        prediction = F.softmax(prediction)
        prediction = prediction[0].cpu().detach().numpy()
        action = np.random.choice(range(prediction.shape[0]), p = prediction)
        return action


