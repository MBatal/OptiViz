import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable
from algorithms.pso.swarm import Swarm, Particle  # Assuming Swarm and Particle classes are defined in the "swarm.py" file in the other folder.
from algorithms.pso.particle import Particle
from visualization.visualizer import SwarmVisualizer  # Assuming SwarmVisualizer is defined in "visualization.py"
from algorithms.fitness import rastrigin, ackley, square, get_spacial_bounds, sphere, rosenbrock

fit_func = rosenbrock
low_bound, hi_bound, _, _ = get_spacial_bounds(fit_func)
swarm = Swarm(dimension=3, lower_bound=low_bound, upper_bound=hi_bound, n_particles=30, fitness_func=fit_func, n_iterations=1000)
best_position, best_fitness = swarm.optimize()
print(f"Best Position: {best_position}, Best Fitness: {best_fitness}")

# Visualize the final state
visualizer = SwarmVisualizer(swarm, fitness_func=fit_func)
visualizer.plot_function()

# NOTE:  python -m testing.animate_test