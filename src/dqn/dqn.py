import numpy as np
from net import Net
import gym


class Agent(object):

    def __init__(self, env_name, gamma, memory_size, 
            exploration_rate, exploration_min, exploration_decay,
            lr, batch_size, is_render = False):

        self.env = gym.make(env_name)
        self.gamma = gamma
        self.memory_size = memory_size
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.lr = lr
        self.batch_size = batch_size
        self.is_render = is_render
        self.action_space = self.env.action_space.n # action_space
        self.state_space = self.env.observation_space.shape[0] # state space
        self.Target_Net = Net()

        self.memory = 

        return


    def memory(self, s, a, r, s_):
        # store current state, action, reward, next state

        return

    def choose_action(self, state):

        return action


    def experience_replay(self):


    def learn(self):


