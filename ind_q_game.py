import numpy as np 
import random
import matplotlib.pyplot as plt 
from game_dynamics.game_setup import GameVariables
from game_dynamics.q_agent import  QAgent
from game_dynamics.epsilon_greedy import EpsilonGreedy
from game_dynamics.ind_q_env import Environment

## simulate two individual traffic lights with q learning 
num_p = 3

p = np.linspace(0.15,0.85,num_p)
q = np.linspace(0.15,0.85,num_p)
env = Environment(p=p, 
                  q=q, 
                  num_players_1=100, 
                  num_players_2=100, 
                  sim_time=150000)
obs = env.reset()
obs = np.around(obs, 1) 

ql_agents = {'TL1': QAgent(starting_state=obs[0],
                                 action_space=np.arange(env.game.num_q),
                                 alpha=0.1,
                                 gamma=0.99,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.8,
                                                                    min_epsilon=0.005,
                                                                    decay=0.99)),
             'TL2': QAgent(starting_state=obs[1],
                                 action_space=np.arange(env.game.num_p),
                                 alpha=0.1,
                                 gamma=0.99,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.8,
                                                                    min_epsilon=0.005,
                                                                    decay=0.99))}


env.reset()
for time in range(env.game.sim_time):

    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

    obs, r, done, _ = env.step(action=actions)
    obs = np.around(obs, 1) 

    ql_agents['TL1'].learn(new_state=obs[0],
                           reward=r['TL1']
                           )
    ql_agents['TL2'].learn(new_state=obs[1], 
                           reward=r['TL2']
                           )








