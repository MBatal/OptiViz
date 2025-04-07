from algorithms.fitness import get_spacial_bounds, rastrigin
from algorithms.pso.swarm import Swarm
from visualization.visualizer import SwarmVisualizer

fit_func = rastrigin
low_bound, hi_bound, _, _ = get_spacial_bounds(fit_func)
swarm = Swarm(
    dimension=3,
    lower_bound=low_bound,
    upper_bound=hi_bound,
    n_particles=30,
    fitness_func=fit_func,
    n_iterations=1000
)
best_position, best_fitness = swarm.optimize()
print(f"Best Position: {best_position}, Best Fitness: {best_fitness}")

# Visualize the final state
visualizer = SwarmVisualizer(swarm, fitness_func=fit_func)
# Visualize the final state zoomed in
# visualizer = SwarmVisualizer(swarm, fitness_func=fit_func, clip=True)
visualizer.plot_function()

# NOTE:  python -m testing.animate_test