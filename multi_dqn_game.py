import numpy as np 
import random
import dqn_agent
from game_dynamics.dqn_agent import DQNAgent
from game_dynamics.dqn_agent import Model 
from game_dynamics.epsilon_greedy import EpsilonGreedy
from game_dynamics.multi_dqn_env import Environment
import matplotlib.pyplot as plt 
from game_dynamics.game_setup import GameVariables
from timeit import default_timer as timer
import random, os.path, math, glob, csv, base64

from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output

num_p = 10

p = np.linspace(0.15,0.85,num_p)
q = np.linspace(0.15,0.85,num_p)

env = Environment(p=p, q=q, num_players_1=100, num_players_2=100, sim_time=150000)

start=timer()

log_dir = "./data/"
try:
    os.makedirs(log_dir)
except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv')) \
        + glob.glob(os.path.join(log_dir, '*td.csv')) \
        + glob.glob(os.path.join(log_dir, '*action_log.csv'))
    for f in files:
        os.remove(f)

config = dqn_agent.Config(sim_time=env.game.sim_time)

model  = Model(env=env, config=config, log_dir=log_dir)

episode_reward = 0

observation = env.reset()
for sample_idx in range(1, config.MAX_SAMPLES + 1):
    
    epsilon = config.epsilon_by_sample(sample_idx)

    action = model.get_action(observation, epsilon)

    model.save_action(action, sample_idx)

    prev_observation=observation
    observation, reward, done, _ = env.step(action)
    observation = None if done else observation

    model.update(prev_observation, action, reward, observation, sample_idx)
    episode_reward += reward


model.save_w()
env.close()



