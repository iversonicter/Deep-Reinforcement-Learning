# Notes in DRL

- value-based optimization:  Q learning, Sarsa, Deep Q Network

- policy-based optimization: Policy gradients

- model-based optimization: Model-based RL

# Categroies in DRL

1. model-free or model-based ?

model-free: passively get the feedback from outside

model-based: can predict what consequence can be achieved given a specific action

model-free: Q-learning, Sarsa, policy gradients

model-based: AlphaGo

2. value-based or policy based? 

value-based drl choose action by the expected maximum profits, not suitable for continous action
policy-based drl: dircetly optimize the policy and evalute the possibility of each action.

valued-based: Q learning, Sarsa, DQN

policy-based: 

hybird valued-based and policy-based: A3C

3. monte-carlo update or Temporal-Difference update

monte-carlo update: monte-carlo learning, orginal policy gradient


TF updates: Q learning, sarsa, advanced policy gradient

4. On-policy or off-policy?

on-policy: Sarsa

off-policy: Q-learning, DQN


# Reasons of instability in DQN

[content source](https://jaromiru.com/2016/10/12/lets-make-a-dqn-debugging/)

## unappropriate network size

small network may fail to approximate the Q function properly, large network can lead to overfitting. careful network tuning of a network size along with other hyper-parameters can help

## Moving targets

Target depends on the current network estimates, which means the target for leraning moves with each training step. Before teh network has a chance to converge to the desired values, the target changes, resulting in possible oscillation or divergence. Solution is to use a fixed values, which are regularly updated.

## Maximization bias

Due to the max in the formula for setting targets, the network suffers from maximization bias, possibly leading to overestimatation of the Q function's value and poor performance. Double learning can help

## outliers with high weight
When a training is performed on a sample which does not correspond to current estimate, it can change the network weights substantially.because of the high loss value in MSE loss function, leading to poor performance. The solution is to use a los clipping or a Hubert Loss function

## biases data in memory
If the memory contain only some set of data, the training on this data can eventually change the learned values for important states, leading to poor performance os oscillations. During learning, the agent chooses actions according to its current policy, filling its memory with biases data. Enough exploration can help.

## Prioritized experience replay

Randomly sampling from a memory is not very efficient. using a more sophisicated sampling strategy can speed up the learning process.

## utilizing known truth
Some values of the Q function are known exactly. These are those at the end of the episode, leading to t he terminal state. Their value is exactly Q(s,a) = r. The idea is to use these values as anchors, possibly with higher weight in the learning step, which the network can hold to.

