from collections import deque
import random
import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, s_1, t):
        experience = (s, a, r, s_1, t)
        if(self.count == self.buffer_size):
            self.buffer.popleft()
            self.buffer.append(experience)
        else:
            self.buffer.append(experience)
            self.count = self.count + 1

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []
        if(batch_size > self.count):
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s_1_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, s_1_batch, t_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

    
    
