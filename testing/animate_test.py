import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable
from algorithms.pso.swarm import Swarm, Particle  # Assuming Swarm and Particle classes are defined in the "swarm.py" file in the other folder.
from algorithms.pso.particle import Particle
from visualization.visualizer import SwarmVisualizer  # Assuming SwarmVisualizer is defined in "visualization.py"
from algorithms.fitness import rastrigin, ackley, square

# Initialize Swarm with 30 particles in 2D space
swarm = Swarm(dimension=3, n_particles=30, lower_bound=-5, upper_bound=5, fitness_func=rastrigin)

# Visualizer
visualizer = SwarmVisualizer(swarm, fitness_func=rastrigin, clip=False)

# Visualize the function and particles in 2D or 3D
visualizer.plot_function()  # You can switch this between plot_2d and plot_3d depending on your dimension

# Animate the PSO process over 100 iterations with a delay of 200ms between frames
visualizer.animate(n_iterations=1000, interval=200)

# NOTE:  python -m testing.animate_test