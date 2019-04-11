import gym
from collections import deque
from util import *
import random
from net import *
from config import *

class Agent(object):

    def __init__(self, env_name, gamma, memory_size, 
            exploration_rate, exploration_min, exploration_decay,
            learning_rate, action_space, state_space, batch_size):

        self.gamma = gamma
        self.memory_size = memory_size
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.batch_size = batch_size
        self.action_space = action_space

        self.QNet = DQN(learning_rate, action_space, state_space)
        self.memory = deque(maxlen = memory_size)

        self.env = gym.make(env_name)

    def act(self, state):
        # 0,1 meanless
        # 2,4 up
        # 3,5 down
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.QNet.forward(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.gamma * np.max(self.QNet.forward(state_next)[0]))
            q_values = self.QNet.forward(state)
            q_values[0][action] = q_update
            self.QNet.backward(state, q_values)

        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)


    def run(self):
        episode = 0
        render = False
        while True:
            episode += 1
            real_state = self.env.reset()
            state = preprocess_atari(real_state)
            state = state[np.newaxis, :]
            step = 0
            #if (episode + 1) % 50 == 0:
            render = True
            while True:
                if render:
                    self.env.render()
                step += 1
                action = self.act(state)
                state_next, reward, terminal, info = self.env.step(action + 1)
                reward = reward if not terminal else -reward
                state_next = preprocess_atari(state_next)
                state_next = state_next[np.newaxis, :]
                self.remember(state, action, reward, state_next, terminal)
                state = state_next

                if terminal:
                    print("Run: " + str(run) + ", exploration: " + str(self.exploration_rate) + ", score" + str(step))
                    break
                self.experience_replay()


if __name__ == "__main__":
    instance = Agent(name, gamma, memory_size, exploration_rate, exploration_min, exploration_decay, lr, 3, state_space = [80, 80, 1], batch_size = 32)
    instance.run()

