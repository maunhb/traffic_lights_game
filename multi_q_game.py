import numpy as np 
import random
import matplotlib.pyplot as plt 
from game_dynamics.game_setup import GameVariables
from game_dynamics.multi_q import MultiQAgent
from game_dynamics.multi_epsilon_greedy import EpsilonGreedy
from game_dynamics.multi_env import Environment

num_p = 3

p = np.linspace(0.15,0.85,num_p)
q = np.linspace(0.15,0.85,num_p)
env = Environment(p, q, 100, 100, 1500)#00)
obs = env.reset()
obs = np.around(obs, 1) 

ql_agents = {'TL': MultiQAgent(starting_state=obs,
                               action_space=[num_p, num_p],
                               alpha=0.1,
                               gamma=0.99,
                               exploration_strategy=EpsilonGreedy(initial_epsilon=0.8,
                                                                  min_epsilon=0.005, 
                                                                  decay=0.99))}

env.reset()
for time in range(env.game.sim_time):

    actions =  ql_agents['TL'].act() 

    obs, r, done, _ = env.step(action=actions)

    ql_agents['TL'].learn(new_state=np.around(obs,1),
                           reward=r['TL']
                           )



