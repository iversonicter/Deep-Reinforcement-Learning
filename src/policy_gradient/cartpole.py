# Author: Wang Yongji
# Email : yongjie.wang@ntu.edu.sg
# Description: cartpole 

import gym
from policy_gradient import Agent
from net import Net

env = gym.make('CartPole-v0')
env.seed(1)
is_render = False

eval_net = Net(env.observation_space.shape[0], env.action_space.n)
eval_net = eval_net.cuda()
agent = Agent(eval_net, lr = 0.02, reward_decay = 0.99)

for i in range(3000):
    observation = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward)
        total_reward += reward
        observation = observation_
        if done:
            agent.learn()
            print("episode: ", i, "\t total reward: ", total_reward)
            break


