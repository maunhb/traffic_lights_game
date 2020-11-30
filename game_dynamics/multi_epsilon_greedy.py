import numpy as np
from gym import spaces
import random

class EpsilonGreedy:

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space):
        action = [0,0]
        if np.random.rand() < self.epsilon:
            action[0] = np.random.choice(action_space[0])
            action[1] = np.random.choice(action_space[1])
        else:
            action[0], action[1] =  self.argmax_2D(q_table['{}'.format(state)])

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        return action

    def reset(self):
        self.epsilon = self.initial_epsilon
    
    # def randargmax(self, b, **kw):
    #     """ a random tie-breaking argmax"""
    #     return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)

    def argmax_2D(self, b):
        ind = np.unravel_index(np.argmax(np.random.random(b.shape) * (b==b.max())), b.shape)
        return ind 
        
