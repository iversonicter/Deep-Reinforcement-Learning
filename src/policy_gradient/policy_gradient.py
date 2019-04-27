# Author: Wang Yongjie
# Email:  yongjie.wang@ntu.edu.sg
# Description: Policy gradient

import numpy as np
import torch

class Agent(object):

    def __init__(self, eval_net, lr, reward_decay = 0.95):

        self.Eval_Network = eval_net # train with GPU
        self.lr = lr
        self.reward_decay = reward_decay
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

        states = torch.from_numpy(np.stack(self.states))
        actions = torch.form_numpy(np.stack(self.actions))
        states  = Variable(states.float().cuda())
        actions = Variable(actions.float().cuda())
        discount_reward = Variable(torch.from_numpy(discount_reward).float().cuda())

        predicition = self.Eval_Network(state)

        loss = self.loss_func(prediction, actions)
        weighted_loss = loss * discount_reward

        # update loss
        self.optim.zero_grad()
        weighted_loss.backward()
        self.optim.step()

