import pickle
import neat
from utils import Ball, Agent, Environment

ball = Ball(10, 0)
agent1 = Agent(-20, ball, 1, damping=1)
agent2 = Agent(5, ball, 0.5, damping=1)

desired_trajectory = [-10] * 25 + [0] * 25 + [10] * 25 + [0] * 25

env = Environment(ball, desired_trajectory, agent1, agent2, debug=True)

obs, _ = env.reset()
done = False
while not done:
    p_coeff = 0.2
    error, ball_vel = obs[0]
    val = p_coeff * error  
    obs, reward, done, _, _ = env.step(val)
    
    env.render()
print("final reward:", reward)
