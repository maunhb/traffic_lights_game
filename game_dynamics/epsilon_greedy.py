import numpy as np
from gym import spaces


class EpsilonGreedy:

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(action_space)
        else:
            #action = np.argmax(q_table['{}'.format(state)])
            action = self.randargmax(np.array(q_table['{}'.format(state)]))
            
        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)

        return action

    def reset(self):
        self.epsilon = self.initial_epsilon

    def randargmax(self, b,**kw):
        """ a random tie-breaking argmax"""
        return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)
