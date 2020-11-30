import numpy as np
from game_dynamics.multi_epsilon_greedy import EpsilonGreedy

class MultiQAgent():

    def __init__(self, starting_state, action_space, alpha=0.5, gamma=0.95,
                 exploration_strategy=EpsilonGreedy()):
        self.state = starting_state
        self.action_space = action_space
        self.action = [1,1]
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {'{}'.format(self.state): -2*np.ones((self.action_space[0], self.action_space[1]))}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self):
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, new_state, reward, done=False):
        if '{}'.format(new_state) not in self.q_table:
            self.q_table['{}'.format(new_state)] = np.zeros((self.action_space[0], self.action_space[1]))

        s = self.state
        s1 = new_state
        self.q_table['{}'.format(s)][self.action[0]][self.action[1]] += \
             self.alpha*(reward + self.gamma*np.max(self.q_table['{}'.format(s1)])-
                         self.q_table['{}'.format(s)][self.action[0]][self.action[1]]
                         )
        self.state = s1
        self.acc_reward += reward
