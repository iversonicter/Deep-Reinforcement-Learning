# Policy gradients

Poliy gradient methods are ubiquitous in model free reinforcment learning algorthms.

In essence, policy gradient methods update the probability distribution of actions so that actions  with higher expected reward have a high probability value for an observed state. 

The objective function for policy gradients is defined as:

<img src="http://latex.codecogs.com/gif.latex?J(\theta)=E\[\sum_{t=1}^{T=1}r_{t+1}\]"/>
