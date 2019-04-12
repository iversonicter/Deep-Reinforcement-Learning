import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random

env_name = "CartPole-v1"
gamma = 0.95
learning_rate = 0.001

memory_size = 1000000
batch_size = 20

exploration_max = 1.0
exploration_min = 0.01
exploration_decay = 0.995

class DQNSolver:
    def __init__(self, observation_space, action_space):

        self.exploration_rate = exploration_max
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen = memory_size)
        self.input_x = tf.placeholder(tf.float32, [None, self.observation_space])
        self.labels = tf.placeholder(tf.float32, [None, self.action_space])
        self.init_tf_env()
        self.fc3 = self.create_network(self.input_x)
        losses = tf.losses.mean_squared_error(labels = self.labels, predictions = self.fc3)
        self.optim = tf.train.AdamOptimizer(learning_rate).minimize(losses)
        self.sess.run(tf.global_variables_initializer())

    def fc(self, x, num_in, num_out, name, relu = True):
        """
        -x:        input tensor
        -num_in:   input length
        -num_out:  output length
        -relu:     default activated function relu
        """
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable("weights", shape = [num_in, num_out],initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1))
            biases = tf.get_variable("biases", shape = [num_out], initializer = tf.constant_initializer(0))
            fc = tf.add(tf.matmul(x, weights), biases, name = scope.name)

            if relu:
                relu = tf.nn.relu(fc)
                return relu
            else:
                return fc

    def init_tf_env(self):

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth= True
        self.sess = tf.Session(config = config)

    def create_network(self, input_x):
        fc1 = self.fc(input_x, self.observation_space, 24, 'fc1', relu = True)
        fc2 = self.fc(fc1, 24, 24, 'fc2', relu = True)
        fc3 = self.fc(fc2, 24, self.action_space, 'fc3', relu = False)
        return fc3

    def predict(self, real_input):
        _fc3 = self.sess.run(self.fc3, feed_dict = {self.input_x:real_input})
        return _fc3

    def update(self, state, target):
        self.sess.run(self.optim, feed_dict = {self.input_x:state, self.labels:target})


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + gamma * np.max(self.predict(state_next)[0]))  # network output
            q_values = self.predict(state) # network output
            q_values[0][action] = q_update
            self.update(state, q_values)

        self.exploration_rate *= exploration_decay
        self.exploration_rate = max(exploration_min, self.exploration_rate)

def cartpole():
    env = gym.make(env_name)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)

    run = 0
    while run < 100:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next,terminal)
            state = state_next

            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                break
            dqn_solver.experience_replay()


    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    terminal = True
    while terminal:
        env.render()
        action = dqn_solver.act(state)
        state_next, reward, terminal, info = env.step(action)


if __name__ == "__main__":
    cartpole()

