# Author : Wang Yongjie
# Email  : yongjie.wang@ntu.edu.sg
# Description: DQN

from net import Net
from memory import Memory
import gym
import torch
from torch.autograd import Variable
import numpy as np
import random
from utils import *


class Prioritized_Double_DQN(object):

    def __init__(self, env_name, gamma, memory_size, replace_iter,
            exploration_rate, exploration_min, exploration_decay,
            lr, batch_size, epoches = 1000, is_render = True):

        self.env = gym.make(env_name)
        self.gamma = gamma
        self.memory_size = memory_size # memroy size
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
        self.state_space = (2, 84, 84) # state space
        self.Eval_Net = Net(self.state_space, self.action_space).cuda()
        self.Target_Net = self.Eval_Net

        self.loss_func = torch.nn.SmoothL1Loss().cuda()
        #self.loss_func = torch.nn.MSELoss().cuda()
        self.optim = torch.optim.RMSprop(self.Eval_Net.parameters(), lr = self.lr)

        self.memory = Memory(memory_size)

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

    def _getTarget(self, batch_samples):

        terminal_state = np.zeros(self.state_space)
        state = np.array([o[1][0] for o in batch_samples]) # current state
        next_state = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch_samples]) # next state
        # convert numpy to tensor
        state = torch.from_numpy(state)
        next_state = torch.from_numpy(next_state)
        state = Variable(state.float().cuda())
        next_state = Variable(next_state.float().cuda())

        # evaluate the current state
        eval_current = self.Eval_Net(state)

        # evaluate the next state
        eval_next = self.Eval_Net(next_state)
        target_next = self.Target_Net(next_state)

        # return variable
        x = np.zeros((len(batch_samples), IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT))
        y = np.zeros((len(batch_samples), self.action_space))
        errors = np.zeros(len(batch_samples))

        for i in range(len(batch_samples)):
            sample = batch_samples[i][1] # batch_samples (errors, sample)
            state, action, reward, next_state = sample[0], sample[1], sample[2], sample[3]

            t = eval_current[i]
            oldVal = t[action]

            if next_state is None:
                t[action] = r
            else:
                action_next = torch.argmax(eval_next[i]).item()
                t[action] = (reward + self.gamma * target_next[i][action_next]) # double DQN

            x[i] = state.datach().numpy()
            y[i] = t.detach().numpy()
            errors[i] = torch.abs(oldVal - t[action])

        return (x, y, errors)

    def memorize(self, samples): 
        # samples format (current state, action, reward, next state)
        x, y, errors = self._getTarget([(0, samples)])
        self.memory.add(errors[0], samples)

    def experience_replay(self):

        if self.learn_step_counter % self.replace_iter == 0:
            self.Target_Net = self.Eval_Net

        batch_samples = self.memory.sample(self.batch_size) # sample for memory
        batch_loss = 0
        x, y, errors = self._getTarget(batch_samples)
        for i in range(len(batch_samples)):
            idx = batch_samples[i][0]
            self.memory.update(idx, errors[i])
            loss = self.loss_func(y[i], y[i] + errors[i])
            batch_loss += loss

        self.optim.zero_grad()
        batch_loss.backward()
        self.optim.step()

        self.exploration_rate = self.exploration_rate * self.exploration_decay
        if self.exploration_rate < self.exploration_min:
            self.exploration_rate = self.exploration_min

        self.learn_step_counter += 1
        self.cost_hist.append(batch_loss.item() / self.batch_size)

    def run(self):
        total_step = 0
        for i in range(self.epoches):
            observation = self.env.reset()
            observation = preprocess(observation)
            current_state = np.array([observation, observation])
            ep_r = 0
            step = 0
            while True:
                if self.is_render:
                    self.env.render()

                action = self.choose_action(current_state)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.array([current_state[1], preprocess(next_state)])
                reward = np.clip(reward, -1, 1) # clip reward to [-1, 1]

                self.memorize((current_state, action, reward, next_state))
                if total_step > 10:
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

    def close(self):
        self.env.close()

