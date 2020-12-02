import numpy as np 
import random
import matplotlib.pyplot as plt 
from game_dynamics.game_setup import GameVariables
from game_dynamics.coord_agent import  CoordAgent 
from game_dynamics.coord_epsilon_greedy import EpsilonGreedy
from game_dynamics.coord_env import Environment
from game_dynamics.variable_elimination import VariableElimination

coord_graph = {
    0:[1],
    1:[0]
}
# remove any duplicates from coord graph
# coord edges represent the edges as a vector where 
coord_edges = []
vertex_list = list(coord_graph.keys())
for vertex in coord_graph:
    for i in range(0,len(coord_graph[vertex])):
        if coord_graph[vertex][i] in vertex_list:
            coord_edges = np.append(coord_edges, vertex)
            coord_edges = np.append(coord_edges, coord_graph[vertex][i])
    vertex_list.remove(vertex)
elim_ordering = [0,1] 

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

coord_agents = {edge: CoordAgent(joint_starting_state=[obs[int(coord_edges[edge])],
                                                       obs[int(coord_edges[edge+1])]],
                            joint_state_space=[env.observation_space, env.observation_space],
                            joint_action_space=[env.action_space,env.action_space],
                            alpha=0.2,
                            gamma=0.9) for edge in range(0,len(coord_edges),2)} 


strategy = EpsilonGreedy(initial_epsilon=0.8, min_epsilon=0.005, decay=0.99)
env.reset()
for time in range(env.game.sim_time):

    q_functions = {edge: coord_agents[edge].q_table['{}'.format(coord_agents[edge].state)] 
                                                             for edge in coord_agents.keys()}

    ve = VariableElimination(q_functions, elim_ordering, coord_edges)
    opt_actions = ve.VariableElimination()
    action_profile = strategy.choose(opt_actions, env.action_space)

    s, r, done, _ = env.step(action=action_profile)
    s = np.around(s, 1) 
    coord_agents[0].learn(new_state=[s[0],s[1]],
                          actions=[action_profile[0],action_profile[1]],
                          reward=r['TL1']+r['TL2']) 










