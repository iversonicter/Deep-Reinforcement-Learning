# Author : Wang Yongjie
# Email  : yongjie.wang@ntu.edu.sg
# Description: DQN

from collections import deque 
from net import Net
import gym
import torch
from torch.autograd import Variable
import numpy as np
import random


class Agent(object):

    def __init__(self, env_name, gamma, memory_size, replace_iter,
            exploration_rate, exploration_min, exploration_decay,
            lr, batch_size, epoches = 1000, is_render = True):

        self.env = gym.make(env_name)
        self.gamma = gamma
        self.memory_size = memory_size # memroy size
        self.memory_counter = 0
        self.learn_step_counter = 0 #update the target network
        self.replace_iter = replace_iter # 
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.lr = lr
        self.epoches = epoches
        self.batch_size = batch_size
        self.is_render = is_render
        self.cost_hist = []
        self.action_space = self.env.action_space.n # action_space
        self.state_space = self.env.observation_space.shape[0] # state space
        self.Eval_Net = Net(self.state_space, self.action_space).cuda()
        self.Target_Net = self.Eval_Net

        self.loss_func = torch.nn.MSELoss().cuda()
        self.optim = torch.optim.SGD(self.Eval_Net.parameters(), lr = self.lr)

        #self.memory = np.zeros((self.memory_size, self.state_space * 2 + 2))# two state +  reward, action
        self.memory = deque(maxlen = self.memory_size)


    def memorize(self, s, a, r, s_, done):
        # store current state, action, reward, next state
        if self.memory_counter > self.memory_size:
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = (s, a, r, s_, done)
        else:
            self.memory.append((s, a, r, s_, done))
        self.memory_counter += 1

    def choose_action(self, state):
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        state = Variable(state.float().cuda())
        if np.random.rand() > self.exploration_rate:
            actions_value = self.Eval_Net(state)[0]
            action = torch.argmax(actions_value).item()
        else:
            action = np.random.randint(0, self.action_space)

        return action

    def experience_replay(self):

        #if len(self.memory) < self.batch_size:
        #    return

        if self.learn_step_counter % self.replace_iter == 0:
            self.Target_Net = self.Eval_Net

        batch_samples = random.sample(self.memory, self.batch_size) # random sample for experiences

        batch_loss = 0

        for state, action, reward, next_state, terminal in batch_samples:
            q_update = torch.tensor([reward])
            q_update = Variable(q_update.float().cuda())
            next_state = torch.from_numpy(next_state)
            next_state = next_state.unsqueeze(0)
            next_state = Variable(next_state.float().cuda())
            if not terminal:
                q_update = (q_update + self.gamma * torch.max(self.Target_Net(next_state)[0]))
            
            state = torch.from_numpy(state)
            state = state.unsqueeze(0)
            state = Variable(state.float().cuda())
            q_values = self.Eval_Net(state)[0][action]
            loss = self.loss_func(q_values, q_update)
            batch_loss += loss
            self.optim.zero_grad()
            loss.backward() # compute the gradient
            self.optim.step() # back proprogate

        self.exploration_rate = self.exploration_rate * self.exploration_decay
        if self.exploration_rate < self.exploration_min:
            self.exploration_rate = self.exploration_min

        self.learn_step_counter += 1
        self.cost_hist.append(batch_loss.item() / self.batch_size)
        return

    def run(self):
        total_step = 0
        for i in range(self.epoches):
            observation = self.env.reset()
            ep_r = 0
            step = 0
            while True:
                if self.is_render:
                    self.env.render()

                action = self.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                reward = reward if not done else -reward
                self.memorize(observation, action, reward, observation_, done)
                if total_step > 100:
                    self.experience_replay()

                ep_r += reward
                if done:
                    print("Epoch: ", i, "\t reward: ", ep_r, "\t total steps: ", step)
                    break

                observation = observation_
                total_step += 1
                step += 1


    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_hist)), self.cost_hist)
        plt.ylabel("Cost")
        plt.xlabel("training steps")
        plt.show()


