import numpy as np 
import random
import matplotlib.pyplot as plt 
from game_dynamics.game_setup import GameVariables
from game_dynamics.multi_q import MultiQAgent
from game_dynamics.multi_epsilon_greedy import EpsilonGreedy
from game_dynamics.multi_env import Environment

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
# s1 = [x2, x8] s2 = [x11, x7] x12??
TL1state = [sum(strategies1[1]) + sum(strategies1[2]) + sum(strategies2[0])
            + sum(strategies2[3]), sum(strategies2[1]) + sum(strategies1[3])]
TL2state = [sum(strategies1[0]) + sum(strategies2[1]) + sum(strategies2[2])
            + sum(strategies1[3]), sum(strategies1[1]) + sum(strategies2[3])]

ql_agents = {'TL': MultiQAgent(starting_state=TL1state+TL2state,
                                 action_space=[game.num_p, game.num_q],
                                 alpha=0.1,
                                 gamma=0.99,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.9,
                                                                    min_epsilon=0.005, 
                                                                    decay=0.99))}

env = Environment(p=p, q=q, sim_time=game.sim_time)

def count_x(strategies1, strategies2):
    x = np.zeros(14)
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

def choose_strategy(strategycost, strategies, num_players):
    if (strategycost[0] <= strategycost[1] and strategycost[0] <= strategycost[2]
              and strategycost[0] <= strategycost[3]):
            strategies[0][i] = 1/num_players
            strategies[1][i] = 0
            strategies[2][i] = 0
            strategies[3][i] = 0
    elif (strategycost[1] <= strategycost[0] and strategycost[1] <= strategycost[2]
        and strategycost[1] <= strategycost[3]):
        strategies[0][i] = 0
        strategies[1][i] = 1/num_players
        strategies[2][i] = 0 
        strategies[3][i] = 0 
    elif (strategycost[2] <= strategycost[1] and strategycost[2] <= strategycost[0]
        and strategycost[2] <= strategycost[3]):
        strategies[0][i] = 0
        strategies[1][i] = 0
        strategies[2][i] = 1/num_players
        strategies[3][i] = 0
    else: 
        strategies[0][i] = 0
        strategies[1][i] = 0
        strategies[2][i] = 0
        strategies[3][i] = 1/num_players
    return strategies

g = open('./data/multiq_nump{}_time{}_players{}.csv'.format(game.num_p,game.sim_time,game.num_players_1),'w')
g.write('time,tl1,tl2,sc,reward\n')
stable = 0
x = np.zeros(14)
oldx = np.zeros(14)
for time in range(game.sim_time):
    for i in range(len(x)):
        oldx[i] = x[i]
    oldtl1 = tl1
    oldtl2 = tl2
    if time%100 == 0:
        print('timestep '+str(time)+' out of '+str(game.sim_time))
    
    x = count_x(strategies1, strategies2)

    actions =  ql_agents['TL'].act() 
    tl1 = q[actions[0]]
    tl2 = p[actions[1]]

    for i in range(game.num_players_1):
        x = count_x(strategies1, strategies2)
        
        strategycost1 = [game.cost_1_1(x,oldtl2,oldtl1), game.cost_1_2(x,oldtl2,oldtl1),
                          game.cost_1_3(x,oldtl2,oldtl1),game.cost_1_4(x,oldtl2,oldtl1)]

        strategies1 = choose_strategy(strategycost1, strategies1, game.num_players_1)
    
    for i in range(game.num_players_2):
        x = count_x(strategies1, strategies2)

        strategycost2 = [game.cost_2_1(x,oldtl2,oldtl1), game.cost_2_2(x,oldtl2,oldtl1),
                          game.cost_2_3(x,oldtl2,oldtl1),game.cost_2_4(x,oldtl2,oldtl1)]
        
        strategies2 = choose_strategy(strategycost2, strategies2, game.num_players_2)

    r = env.step(action=actions,
                 observation=x
                 )
    
    ql_agents['TL'].learn(new_state=[x[1],x[7],x[11],x[6]], # was x 11 not 10 
                           reward=r['TL']
                           )

    ue_pq[0][time] = game.social_cost(x,tl2,tl1)
    ue_pq[4][time] = tl1 # p
    ue_pq[5][time] = tl2  #q 


    g.write('{},{},{},{},{}\n'.format(time,tl1,tl2,ue_pq[0][time],r['TL']))
g.close()


