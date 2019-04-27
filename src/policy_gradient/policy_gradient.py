# Author: Wang Yongjie
# Email:  yongjie.wang@ntu.edu.sg
# Description: Policy gradient

import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F

class Agent(object):

    def __init__(self, eval_net, lr, reward_decay = 0.95):

        self.Eval_Network = eval_net # train with GPU
        self.lr = lr
        self.reward_decay = reward_decay
        self.states, self.actions, self.rewards = [], [], []
        self.loss_func = torch.nn.CrossEntropyLoss().cuda()
        self.optim = torch.optim.Adam(self.Eval_Network.parameters(), lr = self.lr)

    def choose_action(self, state):

        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        state = Variable(state.float().cuda())
        prediction = self.Eval_Network(state)
        prediction = F.softmax(prediction)
        prediction = prediction[0].cpu().detach().numpy()
        action = np.random.choice(range(prediction.shape[0]),p = prediction)
        return action

    def store_transition(self, state, action, reward):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):

        discount_reward = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.reward_decay + self.rewards[t]
            discount_reward[t] = running_add
        # normalize episode rewards
        discount_reward -= np.mean(discount_reward)
        discount_reward /= np.std(discount_reward)
        # convert to tensor variable

        states = torch.from_numpy(np.stack(self.states))
        # convert actions to one-hot vector
        actions = np.zeros((len(self.actions), 2))
        actions[np.arange(len(self.actions)), self.actions] = 1
        actions = torch.from_numpy(actions)
        states  = Variable(states.float().cuda())
        actions = Variable(actions.float().cuda())
        discount_reward = Variable(torch.from_numpy(discount_reward).float().cuda())

        prediction = self.Eval_Network(states)
        prediction = F.softmax(prediction)

        loss = torch.sum(-torch.log(prediction) * actions, dim = 1)

        weighted_loss = loss * discount_reward
        weighted_loss = torch.mean(weighted_loss)

        # update loss
        self.optim.zero_grad()
        weighted_loss.backward()
        self.optim.step()

        # clear current episode
        self.states, self.actions, self.rewards = [], [], []

