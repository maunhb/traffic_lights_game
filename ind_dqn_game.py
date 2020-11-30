import numpy as np 
import random
import dqn_agent
from game_dynamics.dqn_agent import DQNAgent
from game_dynamics.dqn_agent import Model 
from game_dynamics.epsilon_greedy import EpsilonGreedy
from game_dynamics.ind_dqn_env import Environment
import matplotlib.pyplot as plt 
from game_dynamicss.game_setup import GameVariables
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
    files = glob.glob(os.path.join(log_dir, '*.idqnmonitor.csv')) \
        + glob.glob(os.path.join(log_dir, '*idqntd.csv')) \
        + glob.glob(os.path.join(log_dir, '*idqnaction_log.csv'))
    for f in files:
        os.remove(f)

config = dqn_agent.Config(sim_time=env.game.sim_time)

model1  = Model(env=env, config=config, log_dir=log_dir)
model2  = Model(env=env, config=config, log_dir=log_dir)

observations = env.reset()

for sample_idx in range(1, config.MAX_SAMPLES + 1):
    
    epsilon = config.epsilon_by_sample(sample_idx)

    action_1 = model1.get_action(observations[0], epsilon)
    action_2 = model2.get_action(observations[1], epsilon)

    model1.save_action(action_1, sample_idx)
    model2.save_action(action_2, sample_idx)

    prev_observation=observations
    observations, rewards, done, _ = env.step(action_1, action_2)
    observations = None if done else observations


    model1.update(prev_observation[0], action_1, rewards[0], observations[0], sample_idx)
    model2.update(prev_observation[1], action_2, rewards[1], observations[1], sample_idx)

model1.save_w()
model2.save_w()
env.close()



