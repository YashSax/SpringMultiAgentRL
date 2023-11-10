import numpy as np
import pickle
import neat
from utils import Ball, Agent, Environment
import time

config_path = "./config-feedforward"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

with open('./winner', 'rb') as f:
    genome = pickle.load(f)
net = neat.nn.FeedForwardNetwork.create(genome, config)

ball = Ball(10, 0)
agent1 = Agent(-5, ball, 1, damping=1)
agent2 = Agent(5, ball, 1, damping=1)

desired_trajectory = [0] * 25 + [1] * 25 + [2] * 25 + [-2] * 25

env = Environment(ball, desired_trajectory, agent1, agent2, debug=True)

(a1_obs, a2_obs), reward, done = env.reset()
while not done:
    action1 = net.activate(np.array(a1_obs))[0]
    action2 = -1 * net.activate(np.array(a2_obs))[0]
    (a1_obs, a2_obs), reward, done = env.step([action1, action2])
    env.render()
