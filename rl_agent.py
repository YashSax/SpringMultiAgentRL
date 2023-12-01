import pickle
import neat
from utils import Ball, Agent, Environment
from stable_baselines3 import PPO

ball = Ball(10, 0)
agent1 = Agent(-20, ball, 1, damping=1)
agent2 = Agent(5, ball, 1, damping=1)

desired_trajectory = [10 * math.sin(i / 20) for i in range(0, 200)]

env = Environment(ball, desired_trajectory, agent1, agent2, debug=False)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# vec_env = model.get_env()
# obs = vec_env.reset()
# done = False

# while not done:
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)

ball = Ball(10, 0)
agent1 = Agent(-20, ball, 1, damping=1)
agent2 = Agent(5, ball, 0.5, damping=1)

import math
desired_trajectory = [-10] * 25 + [0] * 25 + [10] * 25 + [0] * 25
test_env = Environment(ball, desired_trajectory, agent1, agent2, debug=True)
obs, _ = test_env.reset()
done = False
while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)
    print(reward)
    test_env.render()

# obs, done = env.reset(), False
# while not done:
#     p_coeff = 0.2
#     error, ball_vel = obs
#     val = p_coeff * error  
#     obs, reward, done = env.step(val)
    
#     env.render()
# print("final reward:", reward)
