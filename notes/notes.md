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

