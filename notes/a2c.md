# A2C 

In policy graident algorithm, we update the policy parameter through Monto Carlo updates(taking random samples). THis introduces in inherent high variability in log probabilities and cumulative reward issues, because each trajectories during training can deviate from each other at great degrees.

Consequently, the high variability in log probabilities and cumulative reward values will make noise gradients, and cause unstable learning or policy distribution skewing t o a non-optimal direction.



