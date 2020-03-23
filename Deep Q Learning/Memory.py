import numpy as np
import random
from collections import deque


class Memory:
    def __init__(self, mem_size):
        self.memory = deque(maxlen=mem_size)

    def sample(self, batch_size):
        memory = random.choices(self.memory, k=batch_size)
        return memory

    def add(self, memory):
        # state, action, reward, next_state
        self.memory.append(memory)

    def get_current_capacity(self):
        return len(self.memory)


class PER:
    def __init__(self, mem_size, seed=2, epsilon=0.001, alpha=0.6, beta=0.6):
        np.random.seed(seed)
        self.capacity = mem_size
        self.tree_size = 2*mem_size - 1
        self.tree = np.zeros(self.tree_size)  # sum tree, store the priority of nodes
        self.data_set = [None for _ in range(mem_size)]  # store data correspond to node
        self.pos = self.capacity - 1  # next position to store store the priority of a node
        self.current_capacity = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta

    def add(self, data, error):
        priority = (error + self.epsilon) ** self.alpha
        self.data_set[self.pos - self.capacity + 1] = data
        self.tree[self.pos] = priority

        current_pos = self.pos
        while True:
            parent = (current_pos - 1) // 2
            if parent > 0:
                self.tree[parent] += self.tree[self.pos]
                current_pos = parent
            else:
                if parent == 0:
                    self.tree[parent] += self.tree[self.pos]
                break
        self.pos += 1
        self.current_capacity = max(self.current_capacity, self.pos - self.capacity + 1)
        if self.pos == self.tree_size:
            self.pos = self.capacity - 1  # overwrite old memory if current capacity exceeds max capacity

    def sample(self, batch_size):
        prior_intervals = np.arange(0., self.tree[0] * 1.01,
                                    self.tree[0]/batch_size)
        priority = np.random.uniform(prior_intervals[:-1], prior_intervals[1:])
        samples = []
        is_weight = []
        for i in range(batch_size):
            # get a sample each loop
            pos = 0
            prior = priority[i]
            while pos < self.capacity - 1:  # while not a leaf node
                left = 2 * pos + 1
                right = left + 1
                if prior <= self.tree[left]:
                    pos = left
                else:
                    prior -= self.tree[left]
                    pos = right
            samples.append(self.data_set[pos - self.capacity + 1])
            prob = self.tree[pos] / self.tree[0]
            is_weight.append((1 / (prob * self.current_capacity)) ** (-self.beta))
        return samples, np.array(is_weight)

    def get_current_capacity(self):
        return self.current_capacity
#
# if __name__ == "__main__":
#     mem = PER(100000)
#
#     for i in range(100000):
#         mem.add(i, i)
#
#     print(mem.sample(100))

