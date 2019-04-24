# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg
# Description: subtree structure


import numpy as np

class SubTree:

    pointer = 0

    def __init__(self, capacity):

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        #[------parent node------][------leaves to record priority------]
        #  size:capacity - 1            size: capacity
        self.data = np.zeros(capacity, dtype = object)
        #[----------data frame-------------]
        # size : capacity - 1

    def retrieve(self, idx, s):


        return

    def total(self):
        return self.tree[0]

    def add(self, p, data):

        tree_idx = self.pointer + seld.capacity - 1
        self.data[self.pointer]  = data # updata data frame O(1)
        self.update(tree_idx, p) # update tree frame O(log(N))
        
        self.pointer += 1
        of self.pointer >= self.capacity: # replace when exceed the capacity
            self.pointer = 0

    def update(self, idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        # then propagate the change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s):
        """







        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1 # left node
            cr_idx = cl_idx + 1 # right node

            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]




