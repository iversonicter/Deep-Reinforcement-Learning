# Policy gradients

Poliy gradient methods are ubiquitous in model free reinforcment learning algorthms.

In essence, policy gradient methods update the probability distribution of actions so that actions  with higher expected reward have a high probability value for an observed state. 

The objective function for policy gradients is defined as:

<img src="http://latex.codecogs.com/gif.latex?J(\theta)=E\[\sum_{t=0}^{T=1}r_{t+1}\]"/>

The objective is to learn a plicy that maximize the cumulative future reward to be received starting from any given time t until the terminal time T.

Since this is a maximization problem, we can optimize the policy by taking the gradient ascent with the partial derivative of the objective with respect to the policy parameter <img src="http://latex.codecogs.com/gif.latex?\theta"/>

<img src="http://latex.codecogs.com/gif.latex?\theta \gets \theta + \frac{\patial}{\patial_{\theta}}J(\theta)"/>


