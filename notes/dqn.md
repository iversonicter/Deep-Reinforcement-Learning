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

