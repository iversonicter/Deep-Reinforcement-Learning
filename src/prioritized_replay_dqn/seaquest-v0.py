from agent import Prioritized_Double_DQN

if __name__ == "__main__":

    seaquest = Prioritized_Double_DQN("Seaquest-v0", gamma = 0.95,
            memory_size = 100000, replace_iter = 2000,
            exploration_rate = 1, exploration_min = 0.1,
            exploration_max = 1, lr = 0.002, batch_size = 64,
            is_render = True)
    seaquest.run()
    seaquest.plot()
    seaquest.close()
