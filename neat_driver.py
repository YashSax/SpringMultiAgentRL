import numpy as np
import pickle
import neat
from utils import Ball, Agent, Environment

runs_per_net = 5
generations = 50

def eval_genome(genome, config):
    # create the network based off the config file
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    ball = Ball(10, 0)
    agent1 = Agent(-20, ball, 1, damping=1)
    agent2 = Agent(5, ball, 1, damping=1)

    desired_trajectory = [0] * 25 + [1] * 25 + [2] * 25 + [-2] * 25

    env = Environment(ball, desired_trajectory, agent1, agent2, debug=False)

    for runs in range(runs_per_net):
        (a1_obs, a2_obs), reward, done = env.reset()
        while not done:
            action1 = net.activate(np.array(a1_obs))[0]
            action2 = -1 * net.activate(np.array(a2_obs))[0]
            (a1_obs, a2_obs), reward, done = env.step([action1, action2])
        fitnesses.append(reward)
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def main():
    config_path = "./config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    
    NUM_CPU_CORES = 4
    pe = neat.ParallelEvaluator(NUM_CPU_CORES, eval_genome)
    winner = pop.run(pe.evaluate,generations)

    # Save the winner.
    with open('./winner', 'wb') as f:
        pickle.dump(winner, f)

    # Show winning neural network
    print(winner)

if __name__ == "__main__":
    main()