# Author: Wang Yongjie
# Email:  yongjie.wang@ntu.edu.sg
# Description: Policy gradient

import numpy as np
from net import Net
import torch

class Agent(object):

    def __init__(self, eval_net, lr, reward_decay = 0.95, is_render = False):

        self.Eval_Network = Net(state_space, action_space).cuda() # train with GPU
        self.lr = lr
        self.reward_decay = reward_decay
        self.is_render = is_render
        self.states, self.actions, self.rewards = [], [], []
        self.loss_func = torch.nn.CrossEntropyLoss().cuda()
        self.optim = torch.optim.SGD(self.Eval_Network.parameters(), lr = self.lr)

    def choose_action(self, state):
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        state = Variable(state.float().cuda())
        prediction = self.Eval_Network(state)
        prediction = prediction[0].cpu().detach().numpy()
        action = np.random.choice(prediction.shape, prediction)
        return action

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):

        discount_reward = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discount_reward[t] = running_add
        # normalize episode rewards
        discount_reward -= np.mean(discount_reward)
        discount_reward /= np.std(discount_reward)
        # convert to tensor variable

        loss = self.loss_func(self.)
        weighted_loss = loss * discount_reward

        # update loss
        self.optim.zero_grad()
        weighted_loss.backward()
        self.optim.step()

        





