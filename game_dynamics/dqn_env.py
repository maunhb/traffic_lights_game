import numpy as np
import pandas as pd
from gamesetup import GameVariables
from gym.spaces import Box 

class Environment():

    def __init__(self, p, q, num_players_1, num_players_2, sim_time):
        self.game = GameVariables(num_players_1=100, num_players_2=100, num_p=len(p), num_q=len(q), sim_time=sim_time)
        self.pvector = p
        self.qvector = q

        self.observation_space_shape = Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                           high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                           dtype=np.float32) 
        self.action_space_n = self.game.num_p

    def reset(self):
        self.x = np.zeros(14)
        self.random_strategies()
        self.count_strategies()

        self.p = random.choice(p)
        self.q = random.choice(q)

    def step(self, action):

        for i in range(self.game.num_players_1):
            
            strategycost1 = [self.game.cost_1_1(x,self.p,self.q), self.game.cost_1_2(x,self.p,self.q),
                              self.game.cost_1_3(x,self.p,self.q),self.game.cost_1_4(x,self.p,self.q)]
            self.choose_strategy_1(strategycost1)
            self.count_strategies()

        for i in range(self.game.num_players_2):

            strategycost2 = [self.game.cost_2_1(x,self.p,self.q), self.game.cost_2_2(x,self.p,self.q),
                              self.game.cost_2_3(x,self.p,self.q), self.game.cost_2_4(x,self.p,self.q)]
            self.choose_strategy_2(strategycost2)
            self.count_strategies()
            
        self.p = self.pvector[action['TL2']]
        self.q = self.qvector[action['TL1']]

        reward = self._compute_rewards(self.x)

        return reward

    def _compute_rewards(self, strategies):
        rewards = {}
        rewards['TL1'] = - self.game.cost_tl1(strategies, self.q)
        rewards['TL2'] = - self.game.cost_tl2(strategies, self.p)
        return rewards
    
    def count_strategies(self):

        self.x[0] = sum(self.strategies1[0]) + sum(self.strategies1[1]) + sum(self.strategies1[2]) + sum(self.strategies1[3])
        self.x[1] = sum(self.strategies1[1]) + sum(self.strategies1[2]) + sum(self.strategies2[0]) + sum(self.strategies2[3])
        self.x[2] = sum(self.strategies1[2]) + sum(self.strategies2[0]) + sum(self.strategies2[1]) + sum(self.strategies1[3])
        self.x[3] = sum(self.strategies2[0]) + sum(self.strategies2[1]) + sum(self.strategies2[2])
        self.x[4] = sum(self.strategies2[0]) + sum(self.strategies2[3])
        self.x[5] = sum(self.strategies1[0]) + sum(self.strategies1[3])
        self.x[6] = sum(self.strategies1[1]) + sum(self.strategies2[3])
        self.x[7] = sum(self.strategies2[1]) + sum(self.strategies1[3])
        self.x[8] = sum(self.strategies1[2]) + sum(self.strategies1[3])
        self.x[9] = sum(self.strategies2[2]) + sum(self.strategies2[3])
        self.x[10] = sum(self.strategies2[0]) + sum(self.strategies2[1]) + sum(self.strategies2[2]) + sum(self.strategies2[3])
        self.x[11] = sum(self.strategies1[0]) + sum(self.strategies2[1]) + sum(self.strategies2[2]) + sum(self.strategies1[3])
        self.x[12] = sum(self.strategies1[0]) + sum(self.strategies1[1]) + sum(self.strategies2[2]) + sum(self.strategies2[3])
        self.x[13] = sum(self.strategies1[0]) + sum(self.strategies1[1]) + sum(self.strategies1[2]) + sum(self.strategies1[3])

    def random_strategies(self):
        self.strategies1 = np.zeros((4,self.game.num_players_1))
        randinitstrat = np.random.randint(4, size = self.game.num_players_1)
        for i in range(self.game.num_players_1):
            self.strategies1[randinitstrat[i]][i] = 1/self.game.num_players_1

        self.strategies2 = np.zeros((4,self.game.num_players_2))
        randinitstrat = np.random.randint(4, size = self.game.num_players_2)
        for i in range(game.num_players_2):
            self.strategies2[randinitstrat[i]][i] = 1/self.game.num_players_2

    def choose_strategy_1(self, strategycost1):
        if (strategycost1[0] <= strategycost1[1] and strategycost1[0] <= strategycost1[2]
                and strategycost1[0] <= strategycost1[3]):
                self.strategies1[0][i] = 1/self.game.num_players_1
                self.strategies1[1][i] = 0
                self.strategies1[2][i] = 0
                self.strategies1[3][i] = 0
        elif (strategycost1[1] <= strategycost1[0] and strategycost1[1] <= strategycost1[2]
                and strategycost1[1] <= strategycost1[3]):
            self.strategies1[0][i] = 0
            self.strategies1[1][i] = 1/self.game.num_players_1 
            self.strategies1[2][i] = 0
            self.strategies1[3][i] = 0
        elif (strategycost1[2] <= strategycost1[1] and strategycost1[2] <= strategycost1[0]
            and strategycost1[2] <= strategycost1[3]):
            self.strategies1[0][i] = 0
            self.strategies1[1][i] = 0
            self.strategies1[2][i] = 1/self.game.num_players_1 
            self.strategies1[3][i] = 0
        else: 
            self.strategies1[0][i] = 0
            self.strategies1[1][i] = 0
            self.strategies1[2][i] = 0
            self.strategies1[3][i] = 1/self.game.num_players_1 

    def choose_strategy_2(self, strategycost2):
        if (strategycost2[0] <= strategycost2[1] and strategycost2[0] <= strategycost2[2] and strategycost2[0] <= strategycost2[3]):
                self.strategies2[0][i] = 1/self.game.num_players_2
                self.strategies2[1][i] = 0
                self.strategies2[2][i] = 0
                self.strategies2[3][i] = 0
        elif (strategycost2[1] <= strategycost2[0] and strategycost2[1] <= strategycost2[2] and strategycost2[1] <= strategycost2[3]):
            self.strategies2[0][i] = 0
            self.strategies2[1][i] = 1/self.game.num_players_2
            self.strategies2[2][i] = 0
            self.strategies2[3][i] = 0
        elif (strategycost2[2] <= strategycost2[1] and strategycost2[2] <= strategycost2[0] and strategycost2[2] <= strategycost2[3]):
            self.strategies2[0][i] = 0
            self.strategies2[1][i] = 0
            self.strategies2[2][i] = 1/self.game.num_players_2
            self.strategies2[3][i] = 0
        else:
            self.strategies2[0][i] = 0
            self.strategies2[1][i] = 0
            self.strategies2[2][i] = 0
            self.strategies2[3][i] = 1/self.game.num_players_2



