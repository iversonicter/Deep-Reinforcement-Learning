from dqn import DQN

if __name__ == "__main__":
    cartpole = DQN("CartPole-v1", gamma = 0.95, memory_size = 100000,
            replace_iter = 4, exploration_rate = 1, exploration_min = 0.01, exploration_decay = 0.999, lr = 0.01, batch_size = 20, is_render = True)
    cartpole.run()
    cartpole.plot()
