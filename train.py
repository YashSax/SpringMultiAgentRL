from utils import Ball, Agent, Environment

ball = Ball(10, 0)
agent1 = Agent(-5, ball, 1, damping=1)
agent2 = Agent(20, ball, 1, damping=1)

desired_trajectory = [0] * 25 + [1] * 25 + [2] * 25 + [1] * 25

env = Environment(ball, desired_trajectory, agent1, agent2, debug=False)

(a1_obs, a2_obs), reward, done = env.reset()
position = []
while not done:
    env.render()
    (a1_obs, a2_obs), reward, done = env.step([0, 0])