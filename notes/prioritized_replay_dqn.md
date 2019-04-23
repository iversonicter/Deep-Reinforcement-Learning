Prioritized Experience Replay is one strategy that tires to leverge this fact by changing the sampling distribution.

The main idea is that we prefer transition that does not fit well to our current estimate of the Q function, because these are the transition that we can learn most from.

We can define an error of a sample S = (s, a, r, s') as a distance between the Q(s,a) and its Target T(s):

error  = |Q(s,a) - T(s)|

For DDQN described above, T it would be 
<img src="http://latex.codecogs.com/gif.latex?T(s) = r + \gamma \hat_Q(s', argmax_a Q(s', a))"/>



One of the possible approaches to PER is proportional prioritization. The error is first conveyed to priority using this formula

<img src="http://latex.codecogs.com/gif.latex?p = (error + \epsilon)^a"/>

Epsilon is a small positive constant that ensures that no transition has zero priority. Alpha, <img src="http://latex.codecogs.com/gif.latex? 0 \le \alpha \le 1"/> controls the different between high and low error. it dertimines how much prioritization is used. With alpha = 0, we would get the uniform case.

Priority is translated to  probability of being choose for replay. A sample i has a probability of being picked during the experience replay determined by a formula

<img src="http://latex.codecogs.com/gif.latex? P_i = \frac{p_i}{\sum_k p_k}"/>

The algorithm is simple - during each learning step we will get a batch of samples with this probability distribution and train our network on it. We only need an effective way of storing these priorities and sampling from them.
