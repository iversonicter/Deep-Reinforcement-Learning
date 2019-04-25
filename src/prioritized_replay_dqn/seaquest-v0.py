from agent import Prioritized_Double_DQN

if __name__ == "__main__":

    seaquest = Prioritized_Double_DQN("Seaquest-v0", gamma = 0.95,
            memory_size = 100000, replace_iter = 20,
            exploration_rate = 1, exploration_min = 0.01,
            exploration_decay = 0.999, lr = 0.0002, batch_size = 20,
            is_render = True)
    seaquest.run()
    seaquest.plot()
    seaquest.close()
