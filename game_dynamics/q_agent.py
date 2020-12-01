import numpy as np
from game_dynamics.epsilon_greedy import EpsilonGreedy

class QAgent():

    def __init__(self, starting_state, action_space, alpha=0.1, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.state = starting_state
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {'{}'.format(self.state): np.zeros(len(self.action_space))}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self):
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, new_state, reward, done=False):
        if '{}'.format(new_state) not in self.q_table:
            self.q_table['{}'.format(new_state)] = np.zeros(len(self.action_space))

        s = self.state
        s1 = new_state
        a = self.action
        self.q_table['{}'.format(s)][a] += self.alpha*(reward + self.gamma*max(self.q_table['{}'.format(s1)]) - self.q_table['{}'.format(s)][a])

        self.state = s1
        self.acc_reward += reward
