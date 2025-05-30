from algorithms.fitness import rastrigin
from algorithms.pso.swarm import Swarm
from visualization.visualizer import SwarmVisualizer

# Initialize Swarm with 30 particles in 2D space
swarm = Swarm(
    dimension=3,
    n_particles=30,
    lower_bound=-5,
    upper_bound=5,
    fitness_func=rastrigin
)

# Visualizer
visualizer = SwarmVisualizer(swarm, fitness_func=rastrigin, clip=False)

# Plotting the function in 2D
visualizer.plot_function()