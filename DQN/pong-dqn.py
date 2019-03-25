import gym
import numpy as np
import random
from net import *
from config import *
from collections import deque



def DQN(object):
    def __init__(self, batch_size, state_space, action_space):
        self.batch_size = batch_size
        self.state_space = state_space
        self.action_space = action_space
        # create tf session
        config = tf.ConfigProto()
        config.gpu_options.all_growth = True
        self.sess = tf.Session(config = config)

        with tf.variable_scope("QNetwork"):
            with tf.variable_scope("conv1"):
                weights = 
                biases = 
            with tf.variable_scope("conv2"):
                weights = 
                biases = 
            with tf.variable_scope("fc1"):
                weights = 
                biases = 
            with tf.variable_scope("fc2"):
                weights = 
                biases = 


        with tf.variable_scope("_QNetwork"):
            with tf.variable_scope("conv1"):
                weights = 
                biases = 
            with tf.variable_scope("conv2"):
                weights = 
                biases = 
            with tf.variable_scope("fc1"):
                weights = 
                biases = 
            with tf.variable_scope("fc2"):
                weights = 
                biases = 


class DQN(object):
    def __init__(self, state, action_space):
        self.state = state
        self.action_space = action_space
        self.queue = deque(maxlen = memory_size)
        self.exploration_rate = exploration_max
        self.input_x = tf.placeholder(tf.float32, [None, 80, 80, 1])
        self.output_y = tf.placeholder(tf.float32, [None, 3])

    def create_network(self, input_x):
        

        return fc

    def action(self):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.predict(state)
        return np.argmax(q_values[0])


    def update():
        return


def main():
    env = gym.make(name)
    observation = env.reset()
    run = 0
    prev_state = None
    while run < 1000:
        run += 1
        state = env.reset()
        state = preproc(state)
        cur_state = state - prev_state
        prev_state = state


    
    return

if __name__ == "__main__":
    main()

