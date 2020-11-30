import numpy as np
from game_dynamics.game_setup import GameVariables

class Environment():

    def __init__(self, p, q, sim_time):
        self.game = GameVariables( num_p=len(p), num_q=len(q), sim_time=sim_time)
        self.pvector = p
        self.qvector = q

    def step(self, action, observation):
        self.p = self.pvector[action[0]]
        self.q = self.qvector[action[1]]
        reward = self._compute_rewards(observation)
        return reward

    def _compute_rewards(self, strategies):
        #return self._total_wait_2(strategies)
        return self._total_wait(strategies)

    ### MY REWARD FUNCTIONS 
    def _total_wait(self, strategies):
        rewards = {}
        rewards['TL'] = - self.game.cost_tl(strategies, self.p, self.q)
        return rewards




