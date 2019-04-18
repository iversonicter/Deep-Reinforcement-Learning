import sys
sys.path.insert(0, '../env')
from maze_env import Maze
from saras import Saras


def update():

    for episode in range(100):
        observation = env.reset()
        action = RL.choose_action(str(observation))

        while True:
            env.render()
            observation_, reward, done = env.step(action)

            action_ = RL.choose_action(str(observation_))

            RL.learn(str(observation), action, reward, str(observation_), action_)
            observation = observation_
            action = action_

            if done:
                break

    print("game over")
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = Saras(action_space = list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()

