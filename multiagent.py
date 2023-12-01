import ray
from ray import tune
from utils import Ball, Agent, MultiAgentEnvironment
import math

ball = Ball(10, 0)
agent1 = Agent(-20, ball, 1, damping=1)
agent2 = Agent(5, ball, 1, damping=1)

desired_trajectory = [10 * math.sin(i / 20) for i in range(0, 200)]

env = MultiAgentEnvironment(ball, desired_trajectory, agent1, agent2, debug=False)

# Initialize Ray
ray.init()

# Configuration and Setup for RLlib Trainer
config = {
    "env": env,
    # Specify your desired RL algorithm here, e.g., "PPO", "DQN", etc.
    "multiagent": {
        "policies": {
            # Define policies for each agent here if necessary
            "agent1": (None, env.observation_space, env.action_space, {}),
            "agent2": (None, env.observation_space, env.action_space, {}),
            # Add more agents if needed
        },
        "policy_mapping_fn": lambda agent_id: agent_id,  # Simplest mapping
    },
    # Other hyperparameters and configurations for the chosen algorithm
}

# Train the agents using RLlib
analysis = tune.run(
    "PPO",  # Replace with your desired algorithm
    config=config,
    stop={"training_iteration": 100},  # Define stopping criteria
    checkpoint_at_end=True  # Save a checkpoint at the end of training
)

# Shutdown Ray when done
ray.shutdown()
