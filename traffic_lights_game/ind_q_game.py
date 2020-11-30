import numpy as np 
import random
import matplotlib.pyplot as plt 
from game_dynamics.game_setup import GameVariables
from game_dynamics.q_agent import  QAgent
from game_dynamics.epsilon_greedy import EpsilonGreedy
from game_dynamics.env import Environment

game = GameVariables(num_players_1=100, num_players_2=100, num_p=3, num_q=3, sim_time=150000)
strategies1 = np.zeros((4,game.num_players_1))
randinitstrat = np.random.randint(4, size = game.num_players_1)
for i in range(game.num_players_1):
    strategies1[randinitstrat[i]][i] = 1/game.num_players_1

strategies2 = np.zeros((4,game.num_players_2))
randinitstrat = np.random.randint(4, size = game.num_players_2)
for i in range(game.num_players_2):
    strategies2[randinitstrat[i]][i] = 1/game.num_players_2

p = np.linspace(0.15,0.85,game.num_p)
q = np.linspace(0.15,0.85,game.num_q)

ue_pq = np.zeros((6,game.sim_time))

tl1 = random.choice(p)
tl2 = random.choice(q)
maxtime = game.sim_time
x = np.zeros(14)
def count_x(x, strategies1, strategies2):
    x[0] = sum(strategies1[0]) + sum(strategies1[1]) + sum(strategies1[2]) + sum(strategies1[3])
    x[1] = sum(strategies1[1]) + sum(strategies1[2]) + sum(strategies2[0]) + sum(strategies2[3])
    x[2] = sum(strategies1[2]) + sum(strategies2[0]) + sum(strategies2[1]) + sum(strategies1[3])
    x[3] = sum(strategies2[0]) + sum(strategies2[1]) + sum(strategies2[2])
    x[4] = sum(strategies2[0]) + sum(strategies2[3])
    x[5] = sum(strategies1[0]) + sum(strategies1[3])
    x[6] = sum(strategies1[1]) + sum(strategies2[3])
    x[7] = sum(strategies2[1]) + sum(strategies1[3])
    x[8] = sum(strategies1[2]) + sum(strategies1[3])
    x[9] = sum(strategies2[2]) + sum(strategies2[3])
    x[10] = sum(strategies2[0]) + sum(strategies2[1]) + sum(strategies2[2]) + sum(strategies2[3])
    x[11] = sum(strategies1[0]) + sum(strategies2[1]) + sum(strategies2[2]) + sum(strategies1[3])
    x[12] = sum(strategies1[0]) + sum(strategies1[1]) + sum(strategies2[2]) + sum(strategies2[3])
    x[13] = sum(strategies1[0]) + sum(strategies1[1]) + sum(strategies1[2]) + sum(strategies1[3])
    return x 

x = count_x(x, strategies1, strategies2)
# s1 = [x2, x8] s2 = [x11, x7]
TL1state = [x[1],x[7]]
TL2state = [x[10], x[6]]  # changed to 10 not 11 

## simulate individual traffic lights

ql_agents = {'TL1': QAgent(starting_state=TL1state,
                                 action_space=np.arange(game.num_q),
                                 alpha=0.1,
                                 gamma=0.99,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.9,
                                                                    min_epsilon=0.005,
                                                                    decay=0.99)),
                                 'TL2': QAgent(starting_state=TL2state,
                                 action_space=np.arange(game.num_p),
                                 alpha=0.1,
                                 gamma=0.99,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.9,
                                                                    min_epsilon=0.005,
                                                                    decay=0.99))}

env = Environment(p=p, q=q, sim_time=game.sim_time)

g = open('./data/indq_nump{}_time{}_players{}.csv'.format(game.num_p,game.sim_time,game.num_players_1),'w')
g.write('time,tl1,tl2,sc,r1,r2\n')

oldx = np.zeros(14)
for time in range(game.sim_time):

    print('timestep '+str(time)+' out of '+str(game.sim_time))

    for i in range(game.num_players_1):
        
        strategycost1 = [game.cost_1_1(x,tl2,tl1), game.cost_1_2(x,tl2,tl1),
                        game.cost_1_3(x,tl2,tl1),game.cost_1_4(x,tl2,tl1)]

        if (strategycost1[0] <= strategycost1[1] and strategycost1[0] <= strategycost1[2]
              and strategycost1[0] <= strategycost1[3]):
            strategies1[0][i] = 1/game.num_players_1
            strategies1[1][i] = 0
            strategies1[2][i] = 0
            strategies1[3][i] = 0
        elif (strategycost1[1] <= strategycost1[0] and strategycost1[1] <= strategycost1[2]
               and strategycost1[1] <= strategycost1[3]):
            strategies1[0][i] = 0
            strategies1[1][i] = 1/game.num_players_1
            strategies1[2][i] = 0 
            strategies1[3][i] = 0 
        elif (strategycost1[2] <= strategycost1[1] and strategycost1[2] <= strategycost1[0]
               and strategycost1[2] <= strategycost1[3]):
            strategies1[0][i] = 0
            strategies1[1][i] = 0
            strategies1[2][i] = 1/game.num_players_1
            strategies1[3][i] = 0
        else: 
            strategies1[0][i] = 0
            strategies1[1][i] = 0
            strategies1[2][i] = 0
            strategies1[3][i] = 1/game.num_players_1
        
        x = count_x(x, strategies1, strategies2)
    
    for i in range(game.num_players_2):

        strategycost2 = [game.cost_2_1(x,tl2,tl1), game.cost_2_2(x,tl2,tl1),
                         game.cost_2_3(x,tl2,tl1),game.cost_2_4(x,tl2,tl1)]
        
        if (strategycost2[0] <= strategycost2[1] and strategycost2[0] <= strategycost2[2] and strategycost2[0] <= strategycost2[3]):
            strategies2[0][i] = 1/game.num_players_2
            strategies2[1][i] = 0
            strategies2[2][i] = 0
            strategies2[3][i] = 0
        elif (strategycost2[1] <= strategycost2[0] and strategycost2[1] <= strategycost2[2] and strategycost2[1] <= strategycost2[3]):
            strategies2[0][i] = 0
            strategies2[1][i] = 1/game.num_players_2
            strategies2[2][i] = 0  
            strategies2[3][i] = 0
        elif (strategycost2[2] <= strategycost2[1] and strategycost2[2] <= strategycost2[0] and strategycost2[2] <= strategycost2[3]):
            strategies2[0][i] = 0
            strategies2[1][i] = 0
            strategies2[2][i] = 1/game.num_players_2
            strategies2[3][i] = 0
        else:
            strategies2[0][i] = 0
            strategies2[1][i] = 0
            strategies2[2][i] = 0
            strategies2[3][i] = 1/game.num_players_2

        x = count_x(x, strategies1, strategies2)

    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
    tl1 = q[actions['TL1']]
    tl2 = p[actions['TL2']]

    r = env.step(action=actions,
                 observation=x
                 )
    
    ql_agents['TL1'].learn(new_state=[x[1],x[7]],
                           reward=r['TL1']
                           )
    ql_agents['TL2'].learn(new_state=[x[10], x[6]], # changed to 10 not 11
                           reward=r['TL2']
                           )

    ue_pq[0][time] = game.social_cost(x,tl2,tl1)
    ue_pq[4][time] = tl1 # p
    ue_pq[5][time] = tl2  #q 

    g.write('{},{},{},{},{},{}\n'.format(time,tl1,tl2,ue_pq[0][time],r['TL1'],r['TL2']))
g.close()




