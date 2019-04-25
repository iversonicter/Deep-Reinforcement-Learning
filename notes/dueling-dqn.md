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

and combine these two streams through a special aggregation layer to get an estimate of Q(s,a)

By decoupling the estimation, intuitively our DDQN can learn which state are(or are not) valuable without having to learn the effect of each action at each state(since it's also calculating V(s))

By decoupling we're able to calculate V(s). This is particularly useful for states where their actions do not affect the environment in a relevant way. In this case, it's unnecessary to calculate the value of each action. For instance, moving right or left only matters if there is a risk of collision. And, in most states, the choice of the action has no effect on what happens

For the aggretation layer, we want to generate the q values for each action at that state. But directly combining the two stream together will fall into the issue of identifiability(given Q(s,a), unable to find A(s,a) and V(s). Not being able to find V(s) and A(s,a) given Q(s,a) will be a problem for backprogation(?). To avoid this problem, force the advantage function estimator to have 0 advantage at the choose action.

To do that, subtract the average advantage of all actions possible of the state.

![formulation](https://cdn-images-1.medium.com/max/1200/0*kt9_Z41qxgiI0CDl)

This architecture help acccelerate the training phase. We can calculate the value of a state without calculating the Q(s,a) for each action at that state.It can help us find much more reliable Q values for each action by decoupling the estimation between two streams.

Implementation details:
```
# two streams with the same structure, just two fully connected layer
self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis = 1, keepdims = True))

```




