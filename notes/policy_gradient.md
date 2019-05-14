# Policy gradients

Poliy gradient methods are ubiquitous in model free reinforcment learning algorthms.

In essence, policy gradient methods update the probability distribution of actions so that actions  with higher expected reward have a high probability value for an observed state. 

The objective function for policy gradients is defined as:

<img src="http://latex.codecogs.com/gif.latex?J(\theta)=E\[\sum_{t=0}^{T=1}r_{t+1}\]"/>

r is the reward received by performing action a at state s

The objective is to learn a plicy that maximize the cumulative future reward to be received starting from any given time t until the terminal time T.

Since this is a maximization problem, we can optimize the policy by taking the gradient ascent with the partial derivative of the objective with respect to the policy parameter <img src="http://latex.codecogs.com/gif.latex?\theta"/>

<img src="http://latex.codecogs.com/gif.latex?\theta \gets \theta + \frac{\partial}{\partial_{\theta}}J(\theta)"/>

## Deriving the policy gradient

<img src="http://latex.codecogs.com/gif.latex?\theta \gets \theta + \frac{\partial}{\partial_{\theta}}J(\theta)"/>

The defined objective function <img src="http://latex.codecogs.com/gif.latex?J(\theta)"/>. Expand the expectation as:

<img src="http://latex.codecogs.com/gif.latex? J(\theta) = E\[\sum_{i=0}^T r_{t+1}|\pi_{\theta}\] = \sum_{t=i}^{T-1}P(s_t, a_t|\tau)r_{t+1}"/>
