import gym
from gym import spaces
import pygame
import numpy as np
import time

def bound(val, min_val, max_val):
    return max(min_val, min(val, max_val))

class Ball:
    def __init__(self, mass, initial_position):
        self.mass = mass
        self.velocity = 0
        self.acceleration = 0
        self.initial_position = initial_position
        self.position = initial_position
    
    def reset(self):
        self.velocity = 0
        self.acceleration = 0
        self.position = self.initial_position
    
    def move(self, force):
        self.acceleration = force / self.mass
        self.velocity += self.acceleration
        self.position += self.velocity

class Agent:
    def __init__(self, initial_position, connected_ball, k, damping=0.5):
        self.initial_position = initial_position
        self.position = initial_position
        self.connected_ball = connected_ball
        self.k = k
        self.damping = damping
        self.max_step_distance = 1 # PLEASE don't change this for now! :)
    
    def move(self, magnitude):
        self.position += bound(magnitude, -self.max_step_distance, self.max_step_distance)

    def get_force_applied(self):
        original_force = -1 * self.k * (self.connected_ball.position - self.position)
        damping = self.connected_ball.velocity * self.damping
        return original_force - damping
    
    def reset(self):
        self.position = self.initial_position

class Environment(gym.Env):
    def __init__(self, ball, desired_trajectory, agent1, agent2, debug=False):
        super(Environment, self).__init__()
        self.ball = ball
        self.agent1 = agent1
        self.agent2 = agent2
        self.desired_trajectory = desired_trajectory
        self.curr_timestep = 0
        self.debug = debug

        obs_low = [-float('inf')] * 4
        obs_high = [float('inf')] * 4
        self.observation_space = spaces.Box(low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32)
        
        agent1_high, agent1_low = agent1.max_step_distance, -agent1.max_step_distance
        agent2_high, agent2_low = agent2.max_step_distance, -agent2.max_step_distance
        action_highs = [agent1_high, agent2_high]
        action_lows = [agent1_low, agent2_low]
        self.action_space = spaces.Box(low=np.array(action_lows), high=np.array(action_highs), dtype=np.float32)

        if debug:
            pygame.init()
            self.width = 800
            self.height = 600
            self.screen = pygame.display.set_mode((self.width, self.height))

    def reset(self):
        self.curr_timestep = 0
        self.ball.reset()
        self.agent1.reset()
        self.agent2.reset()
        
        a1_obs = np.array([self.desired_trajectory[self.curr_timestep], self.ball.position, self.agent1.position, self.agent2.position])
        a2_obs = np.array([self.desired_trajectory[self.curr_timestep], self.ball.position, self.agent2.position, self.agent1.position])
        reward = -1 * abs(self.desired_trajectory[self.curr_timestep] - self.ball.position)
        done = self.curr_timestep >= len(self.desired_trajectory) - 1

        return (a1_obs, a2_obs), reward, done
    
    def __repr__(self):
        return f"Ball = {self.ball.position}, agent1 = {self.agent1.position}, agent2 = {self.agent2.position}"

    def step(self, agent_moves):
        # Returns (observation, reward, done)
        assert len(agent_moves) == 2, "Number of moves != 2"

        self.agent1.move(agent_moves[0])
        self.agent2.move(agent_moves[1])
        
        total_force = self.agent1.get_force_applied() + self.agent2.get_force_applied()
        self.ball.move(total_force)

        desired_position = self.desired_trajectory[self.curr_timestep]
        a1_obs = np.array([desired_position, self.ball.position, self.agent1.position, self.agent2.position])
        a2_obs = np.array([desired_position, self.ball.position, self.agent2.position, self.agent1.position])
        reward = -1 * abs(self.ball.position - desired_position)
        self.curr_timestep += 1
        return (a1_obs, a2_obs), reward, self.curr_timestep >= len(self.desired_trajectory) - 1

    def render(self):
        assert self.debug, "Cannot render if Environment was initialized with debug=False"
        time.sleep(0.1)
        self.screen.fill((255, 255, 255))

        agent1_adjusted = self.width / 2 + 10 * self.agent1.position
        agent2_adjusted = self.width / 2 + 10 * self.agent2.position
        ball_adjusted = self.width / 2 + 10 * self.ball.position
        desired_adjusted = self.width / 2 + 10 * self.desired_trajectory[self.curr_timestep]

        pygame.draw.line(self.screen, (127, 0, 127), (agent1_adjusted, self.height / 2), (ball_adjusted, self.height / 2))
        pygame.draw.line(self.screen, (0, 127, 0), (agent2_adjusted, self.height / 2), (ball_adjusted, self.height / 2))
        pygame.draw.circle(self.screen, (255, 0, 0), (agent1_adjusted, self.height / 2), 15)
        pygame.draw.circle(self.screen, (255, 0, 0), (agent2_adjusted, self.height / 2), 15)
        pygame.draw.circle(self.screen, (0, 255, 0), (ball_adjusted, self.height / 2), 5)
        pygame.draw.circle(self.screen, (0, 0, 0), (desired_adjusted, self.height / 2), 5)
        pygame.display.update()
