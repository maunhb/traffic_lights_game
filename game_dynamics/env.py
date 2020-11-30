import numpy as np
import pandas as pd
from game_dynamics.game_setup import GameVariables

class Environment():

    def __init__(self, p, q, sim_time):
        self.game = GameVariables( num_p=len(p), num_q=len(q), sim_time=sim_time)
        self.pvector = p
        self.qvector = q

    def step(self, action, observation):
        self.p = self.pvector[action['TL1']]
        self.q = self.qvector[action['TL2']]
        reward = self._compute_rewards(observation)
        return reward

    def _compute_rewards(self, strategies):
        rewards = {}
        rewards['TL1'] = - self.game.cost_tl1(strategies, self.q)
        rewards['TL2'] = - self.game.cost_tl2(strategies, self.p)
        return rewards







