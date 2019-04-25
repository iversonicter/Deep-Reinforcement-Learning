# Dueling DQN(DDQN)

Theory: remember that Q-values correspond to how good it is to be at that state and taking an action at that state Q(s,a)

So we can decompose Q(s,a) as the sum of:

- V(s): the value of being at that state
- A(s,a): the advantage of taking that action at that state (how much better is to take this action versus all other possible action at that state)

Q(s, a) = A(s, a) + V(s)

With DDQN, we want to separate the estimator of these two elememts, using two new streams
- One that estimate the state values V(s)
- one that estimate the advantage for each action A(s,a)

![network structure](https://cdn-images-1.medium.com/max/1200/1*FkHqwA2eSGixdS-3dvVoMA.png)


