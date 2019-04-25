# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg
# Description: memory class

from subtree import SubTree
import random

class Memory(object):

    epsilon = 0.01
    alpha = 0.6

    def __init__(self, capacity):
        self.tree =  SubTree(capacity)

    def _getPriority(self, error):
        return (error + self.epsilon) ** self.alpha

    def add(self, error, transition):
        p = self._getPriority(error)
        self.tree.add(p, transition) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (1 + i)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

