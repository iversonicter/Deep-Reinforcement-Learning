from dueling_dqn import Dueling_DQN

if __name__ == "__main__":
    cartpole = Dueling_DQN("CartPole-v1", gamma = 0.95, memory_size = 100000,
            replace_iter = 20, exploration_rate = 1, exploration_min = 0.1, exploration_decay = 0.999, lr = 0.001, batch_size = 20, is_render = True)
    cartpole.run()
    cartpole.plot()
