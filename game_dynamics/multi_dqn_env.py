import numpy as np
import pandas as pd
from game_dynamics.game_setup import GameVariables
from gym import spaces 
import random 

class Environment():

    def __init__(self, p, q, num_players_1, num_players_2, sim_time):
        self.game = GameVariables(num_players_1=100, num_players_2=100, num_p=len(p), num_q=len(q), sim_time=sim_time)
        self.pvector = p
        self.qvector = q

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0]),
                                           high=np.array([1.0, 1.0, 1.0, 1.0]),
                                           dtype=np.float32
                                           ) 

        self.action_space = spaces.Discrete(self.game.num_p*self.game.num_q)

    def close(self):
        self.file.close()

    def reset(self):
        self.x = np.zeros(14)
        self.random_strategies()
        self.count_strategies()

        self.p = random.choice(self.pvector)
        self.q = random.choice(self.qvector)

        self.sim_step  = 0

        self.file = open('./data/multidqn_nump_{}_numplayers_{}_simtime_{}.csv'.format(self.game.num_p,self.game.num_players_1, self.game.sim_time),'w')
        self.file.write('time,tl1,tl2,sc,reward\n')

        return [self.x[1],self.x[7],self.x[6],self.x[11]]

    def step(self, action):

        for i in range(self.game.num_players_1):
            
            strategycost1 = [self.game.cost_1_1(self.x,self.p,self.q), self.game.cost_1_2(self.x,self.p,self.q),
                              self.game.cost_1_3(self.x,self.p,self.q),self.game.cost_1_4(self.x,self.p,self.q)]
            self.choose_strategy_1(i, strategycost1)
            self.count_strategies()

        for i in range(self.game.num_players_2):

            strategycost2 = [self.game.cost_2_1(self.x,self.p,self.q), self.game.cost_2_2(self.x,self.p,self.q),
                              self.game.cost_2_3(self.x,self.p,self.q), self.game.cost_2_4(self.x,self.p,self.q)]

            self.choose_strategy_2(i, strategycost2)
            self.count_strategies()
            
        
        self.p = self.pvector[action%self.game.num_p]
        self.q = self.qvector[int(np.floor(action/self.game.num_p))]
        self.sim_step += 1

        print('sim time '+str(self.sim_step))

        reward = self._compute_rewards(self.x)
        # if self.sim_step > self.game.sim_time:
        #     done = False
        # else:
        #     done = True
        self.collect_data(reward)
 
        return [self.x[1],self.x[7],self.x[6],self.x[11]] , reward, False, {}

    def collect_data(self, reward):
        sc = self.game.social_cost(self.x,self.p,self.q)
        #time,tl1,tl2,sc,reward
        self.file.write('{},{},{},{},{}\n'.format(self.sim_step,self.q,self.p,sc,reward))

    def _compute_rewards(self, strategies):

        rewards = - self.game.cost_tl(strategies, self.p, self.q)
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
        for i in range(self.game.num_players_2):
            self.strategies2[randinitstrat[i]][i] = 1/self.game.num_players_2

    def choose_strategy_1(self, i, strategycost1):
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

    def choose_strategy_2(self, i, strategycost2):
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



