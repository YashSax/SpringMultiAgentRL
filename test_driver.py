import pickle
import neat
from utils import Ball, Agent, Environment

ball = Ball(10, 0)
agent1 = Agent(-20, ball, 1, damping=1)
agent2 = Agent(5, ball, 1, damping=1)

desired_trajectory = [-10] * 25 + [0] * 25 + [10] * 25 + [0] * 25

env = Environment(ball, desired_trajectory, agent1, agent2, debug=True)

(a1_obs, a2_obs), reward, done = env.reset()

while not done:
    desired, curr = a2_obs

    p_coeff = 0.2
    error = curr - desired
    val = -1 * (p_coeff * error)

    # print(f"Curr: {curr}, Desired: {desired}, val: {val}")
    print(f"Ball Velocity: {ball.velocity}")
    
    (a1_obs, a2_obs), reward, done = env.step([val, val])
    env.render()
print("final reward:", reward)
