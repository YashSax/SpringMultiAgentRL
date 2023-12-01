import gymnasium as gym
from gymnasium.spaces.box import Box
import numpy as np
import time
import pygame


def bound(val, min_val, max_val):
    return max(min_val, min(val, max_val))

class Binner:
    def __init__(self, min_val, max_val, num_bins):
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins
    
    def get_bin_from_val(self, val):
        bin_num = int((val - self.min_val) / (self.max_val - self.min_val) * self.num_bins)
        bin_num -= bin_num == self.num_bins
        return bin_num
    
    def get_val_from_bin(self, bin):
        bin_size = (self.max_val - self.min_val) / self.num_bins
        return self.min_val + bin_size * (bin + 0.5)

    def get_states(self):
        return np.array(range(self.num_bins))

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
        self.max_step_distance = 1  # PLEASE don't change this for now! :)

    def move(self, magnitude):
        self.position += bound(
            magnitude, -self.max_step_distance, self.max_step_distance
        )

    def get_force_applied(self):
        original_force = -1 * self.k * (self.connected_ball.position - self.position)
        damping = self.connected_ball.velocity * self.damping
        return original_force - damping

    def reset(self):
        self.position = self.initial_position


class GameState:
    def __init__(self, timestep, ball_position, ball_velocity, agent1_position, agent2_position, desired_position):
        self.timestep = timestep
        self.ball_position = ball_position
        self.ball_velocity = ball_velocity
        self.agent1_position = agent1_position
        self.agent2_position = agent2_position
        self.desired_position = desired_position
    
    def get_q_learning_state(self, position_diff_binner, ball_velocity_binner):
        position_diff = self.ball_position - self.desired_position
        ball_velocity = self.ball_velocity

        position_diff_bin = position_diff_binner.get_bin_from_val(position_diff)
        ball_velocity_bin = ball_velocity_binner.get_bin_from_val(ball_velocity)
        return position_diff_bin, ball_velocity_bin


class Environment(gym.Env):
    def __init__(self, ball, desired_trajectory, agent1, agent2, debug=False):
        super(Environment, self).__init__()
        self.ball = ball
        self.agent1 = agent1
        self.agent2 = agent2
        self.desired_trajectory = desired_trajectory
        self.curr_timestep = 0
        self.debug = debug
        self.cumulative_reward = 0

        obs_low = [-float("inf")] * 2
        obs_high = [float("inf")] * 2
        self.observation_space = Box(
            low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32
        )

        agent1_high, agent1_low = agent1.max_step_distance, -agent1.max_step_distance
        agent2_high, agent2_low = agent2.max_step_distance, -agent2.max_step_distance
        action_highs = [agent1_high]
        action_lows = [agent1_low]
        self.action_space = Box(
            low=np.array(action_lows), high=np.array(action_highs), dtype=np.float32
        )

        if debug:
            pygame.init()
            self.width = 800
            self.height = 600
            self.screen = pygame.display.set_mode((self.width, self.height))

    def get_reward(self):
        reward = -1 * abs(
            self.desired_trajectory[self.curr_timestep] - self.ball.position
        )
        return reward

    def reset(self, seed=1):
        self.curr_timestep = 0
        self.ball.reset()
        self.agent1.reset()
        self.agent2.reset()

        a1_obs = np.array(
            [self.desired_trajectory[self.curr_timestep] - self.ball.position, self.ball.velocity]
        )

        return a1_obs.reshape((1, -1)), {}

    def __repr__(self):
        return f"Ball = {self.ball.position}, agent1 = {self.agent1.position}, agent2 = {self.agent2.position}"

    def get_state(self):
        return GameState(
            timestep=self.curr_timestep,
            ball_position=self.ball.position,
            ball_velocity=self.ball.velocity,
            agent1_position=self.agent1.position,
            agent2_position=self.agent2.position,
            desired_position=self.desired_trajectory[self.curr_timestep]
        )

    def load_state(self, state):
        self.curr_timestep = state.timestep
        self.ball.position = state.ball_position
        self.ball.velocity = state.ball_velocity
        self.agent1.position = state.agent1_position
        self.agent2.position = state.agent2_position

    def step(self, agent_move):
        # Returns (observation, reward, done)
        if type(agent_move) is np.ndarray:
            agent_move = agent_move.item(0)

        self.agent1.move(agent_move)
        self.agent2.move(agent_move)

        total_force = self.agent1.get_force_applied() + self.agent2.get_force_applied()
        self.ball.move(total_force)

        desired_position = self.desired_trajectory[self.curr_timestep]
        a1_obs = np.array([desired_position - self.ball.position, self.ball.velocity])
        reward = -1 * abs(self.ball.position - desired_position)
        self.cumulative_reward += reward
        self.curr_timestep += 1
        return (
            a1_obs.reshape(1, -1),
            reward,
            self.curr_timestep >= len(self.desired_trajectory) - 1,
            False,
            {}
        )

    def render(self):
        assert (
            self.debug
        ), "Cannot render if Environment was initialized with debug=False"
        time.sleep(0.1)
        self.screen.fill((255, 255, 255))

        agent1_adjusted = self.width / 2 + 10 * self.agent1.position
        agent2_adjusted = self.width / 2 + 10 * self.agent2.position
        ball_adjusted = self.width / 2 + 10 * self.ball.position
        desired_adjusted = (
            self.width / 2 + 10 * self.desired_trajectory[self.curr_timestep]
        )

        pygame.draw.line(
            self.screen,
            (127, 0, 127),
            (agent1_adjusted, self.height / 2),
            (ball_adjusted, self.height / 2),
        )
        pygame.draw.line(
            self.screen,
            (0, 127, 0),
            (agent2_adjusted, self.height / 2),
            (ball_adjusted, self.height / 2),
        )
        pygame.draw.circle(
            self.screen, (255, 0, 0), (agent1_adjusted, self.height / 2), 15
        )
        pygame.draw.circle(
            self.screen, (255, 0, 0), (agent2_adjusted, self.height / 2), 15
        )
        pygame.draw.circle(
            self.screen, (0, 255, 0), (ball_adjusted, self.height / 2), 5
        )
        pygame.draw.circle(
            self.screen, (0, 0, 0), (desired_adjusted, self.height / 2), 5
        )
        pygame.display.update()


class MultiAgentEnvironment(gym.Env):
    def __init__(self, ball, desired_trajectory, agent1, agent2, debug=False):
        super(MultiAgentEnvironment, self).__init__()
        self.ball = ball
        self.agent1 = agent1
        self.agent2 = agent2
        self.desired_trajectory = desired_trajectory
        self.curr_timestep = 0
        self.debug = debug
        self.cumulative_reward = 0

        obs_low = [-float("inf")] * 2
        obs_high = [float("inf")] * 2
        self.observation_space = Box(
            low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32
        )

        agent1_high, agent1_low = agent1.max_step_distance, -agent1.max_step_distance
        agent2_high, agent2_low = agent2.max_step_distance, -agent2.max_step_distance
        action_highs = [agent1_high]
        action_lows = [agent1_low]
        self.action_space = Box(
            low=np.array(action_lows), high=np.array(action_highs), dtype=np.float32
        )

        if debug:
            pygame.init()
            self.width = 800
            self.height = 600
            self.screen = pygame.display.set_mode((self.width, self.height))

    def get_reward(self):
        reward = -1 * abs(
            self.desired_trajectory[self.curr_timestep] - self.ball.position
        )
        return reward

    def reset(self, seed=1):
        self.curr_timestep = 0
        self.ball.reset()
        self.agent1.reset()
        self.agent2.reset()

        a1_obs = np.array(
            [self.desired_trajectory[self.curr_timestep] - self.ball.position, self.ball.velocity]
        )

        return a1_obs.reshape((1, -1)), {}

    def __repr__(self):
        return f"Ball = {self.ball.position}, agent1 = {self.agent1.position}, agent2 = {self.agent2.position}"

    def get_state(self):
        return GameState(
            timestep=self.curr_timestep,
            ball_position=self.ball.position,
            ball_velocity=self.ball.velocity,
            agent1_position=self.agent1.position,
            agent2_position=self.agent2.position,
            desired_position=self.desired_trajectory[self.curr_timestep]
        )

    def load_state(self, state):
        self.curr_timestep = state.timestep
        self.ball.position = state.ball_position
        self.ball.velocity = state.ball_velocity
        self.agent1.position = state.agent1_position
        self.agent2.position = state.agent2_position

    def step(self, agent_moves):
        agent1_move, agent2_move = agent_moves
        # Returns (observation, reward, done)
        if type(agent1_move) is np.ndarray:
            agent1_move = agent1_move.item(0)
        if type(agent2_move) is np.ndarray:
            agent2_move = agent2_move.item(0)

        self.agent1.move(agent1_move)
        self.agent2.move(agent2_move)

        total_force = self.agent1.get_force_applied() + self.agent2.get_force_applied()
        self.ball.move(total_force)

        desired_position = self.desired_trajectory[self.curr_timestep]
        a1_obs = np.array([desired_position - self.ball.position, self.ball.velocity])
        reward = -1 * abs(self.ball.position - desired_position)
        self.cumulative_reward += reward
        self.curr_timestep += 1
        return (
            a1_obs.reshape(1, -1),
            reward,
            self.curr_timestep >= len(self.desired_trajectory) - 1,
            False,
            {}
        )

    def render(self):
        assert (
            self.debug
        ), "Cannot render if Environment was initialized with debug=False"
        time.sleep(0.1)
        self.screen.fill((255, 255, 255))

        agent1_adjusted = self.width / 2 + 10 * self.agent1.position
        agent2_adjusted = self.width / 2 + 10 * self.agent2.position
        ball_adjusted = self.width / 2 + 10 * self.ball.position
        desired_adjusted = (
            self.width / 2 + 10 * self.desired_trajectory[self.curr_timestep]
        )

        pygame.draw.line(
            self.screen,
            (127, 0, 127),
            (agent1_adjusted, self.height / 2),
            (ball_adjusted, self.height / 2),
        )
        pygame.draw.line(
            self.screen,
            (0, 127, 0),
            (agent2_adjusted, self.height / 2),
            (ball_adjusted, self.height / 2),
        )
        pygame.draw.circle(
            self.screen, (255, 0, 0), (agent1_adjusted, self.height / 2), 15
        )
        pygame.draw.circle(
            self.screen, (255, 0, 0), (agent2_adjusted, self.height / 2), 15
        )
        pygame.draw.circle(
            self.screen, (0, 255, 0), (ball_adjusted, self.height / 2), 5
        )
        pygame.draw.circle(
            self.screen, (0, 0, 0), (desired_adjusted, self.height / 2), 5
        )
        pygame.display.update()
