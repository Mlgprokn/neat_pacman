from sys import winver
import retro
import numpy as np
import cv2
import neat

def eval_genomes(genomes, config):
    # Track the max fitness from the current run
    max_fitness = 0

    for genome_id, genome in genomes:
        # Set up enviroment variables
        ob = env.reset()
        inx, iny, inc = env.observation_space.shape
        inx = int(inx/8) 
        iny = int(iny/8) 
        
        # Set upt the neural network
        net = neat.nn.RecurrentNetwork.create(genome=genome, config=config)
        
        # Track the fitness for the curent genome and whether the current genome is done
        current_fitness = 0
        done = False
        
        # Game loop
        while not done:
            env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            imgs = []
            imgs = np.ndarray.flatten(ob)
            nnOutput = net.activate(imgs)
            ob, rew, done, info = env.step(nnOutput)
            
            # Update the current fitness to the new points collected and update the lives
            current_fitness += rew
            lives = info['lives']

            # Update the max_fitness
            if current_fitness > max_fitness:
                max_fitness = current_fitness
            
            # If the genome dies once, go to the next genome in line
            if lives < 3:
                done = True    
            
            #x_pos_prev = x_pos
            genome.fitness = current_fitness
            
        # Print the genome id and its fitness
        print(genome_id, current_fitness)
            
# Setup the enviroument and the neat configuration
env = retro.make('PacManNamco-nes')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feetforward')

# Setup the population
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

# Run the game
p.run(eval_genomes)