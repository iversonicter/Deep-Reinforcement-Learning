# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg
# Description: main function


from actor import *
from critic import *
import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env.seed(1)
    is_render = False

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    anet = ANet(state_space, action_space)
    cnet = CNet(state_space)

    actor = Actor(anet, lr = 0.001)
    critic = Critic(cnet, lr = 0.01, gamma = 0.9)

    for i in range(100):
        state = env.reset()
        t = 0
        track_r = []
        while True:
            if is_render: env.render()
            action = actor.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            if done:
                reward = -20
            track_r.append(reward)

            td_error = critic.learn(state, reward, next_state)
            actor.learn(state, action, td_error)

            state = next_state
            t += 1
            if done:
                ep_rs_sum = sum(track_r)
                if "running_reward" not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * .05

                print("episode:\t", i, "\treward\t", running_reward)
                break
            




