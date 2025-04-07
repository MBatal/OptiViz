from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from algorithms.fitness import get_spacial_bounds


class SwarmVisualizer:
    def __init__(self, swarm, fitness_func: Callable, clip: bool=False):
        """
        Initialize SwarmVisualizer

        Args:
            swarm (Swarm): Swarm object used for optimization
            fitness_func (Callable): Fitness function being optimized
            clip (bool): Whether or not to use zoomed in plot (default False)
        """
        self.swarm = swarm
        self.fitness_func = fitness_func
        self.clip = clip

    def plot_function(self):
        if self.swarm.dimension == 2:
            self.plot_2d()
        elif self.swarm.dimension == 3:
            self.plot_3d()

    def plot_2d(self):
        low_bound, hi_bound, _, _ = get_spacial_bounds(fitness_func=self.fitness_func)
        x_vals = np.linspace(low_bound, hi_bound, 100)
        y_vals = np.linspace(low_bound, hi_bound, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array(
            [self.fitness_func(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]
        )
        Z = Z.reshape(X.shape)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)

        # Plot final positions of particles
        final_positions = self.swarm.history[-1]
        ax.scatter(final_positions[:, 0], final_positions[:, 1], c='blue', label="Particles")

        # Plot global best position
        ax.scatter(self.swarm.global_best_position[0], self.swarm.global_best_position[1],
                   c='red', marker='*', s=200, label="Global Best")

        # Create global best star for animation
        global_best_star = ax.scatter(
            [], [], c='yellow', marker='*', s=200, label="Global Best (Star)"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Fitness Landscape with Particles")
        ax.legend()

        # Animation update function
        def update(frame):
            # Get global best position at current frame
            global_best_pos = self.swarm.global_best_history[frame]
            global_best_star.set_offsets([global_best_pos[0], global_best_pos[1]])

            return global_best_star,

        # Create animation
        _ani = FuncAnimation(
            fig, update,
            frames=len(self.swarm.global_best_history),
            interval=500,
            blit=True
        )

        plt.show()

    def plot_3d(self):
        """
        Plot 3D visualization of the function and animate the particles.
        """
        # Get bounds to determine size of plot
        low_bound, hi_bound, _, _ = get_spacial_bounds(fitness_func=self.fitness_func)
        x_vals = np.linspace(low_bound, hi_bound, 100)
        y_vals = np.linspace(low_bound, hi_bound, 100)
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array(
            [self.fitness_func(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]
        )
        Z = Z.reshape(X.shape)
        if self.clip:
            Z = np.clip(Z, 0, np.percentile(Z, 15))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot fitness function as surface
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)

        # Create scatter plot for particles
        scatter = ax.scatter(
            [], [], [], c='deepskyblue', label="Particles", alpha=0.7
        )

        # Create global best star(static)
        global_best_star = ax.scatter(
            [], [], [], c='crimson', marker='*', s=300, label="Global Best"
        )

        # TODO: Add support for optimal solution
        # ax.scatter(0,0,0, c='blue', marker='X', s=150, label="Optimal Solution")

        # Update function for animation
        def update(frame):
            # Extract positions of particles at current iteration
            positions = np.array(self.swarm.history[frame])

            x_vals = positions[:, 0]
            y_vals = positions[:, 1]
            z_vals = positions[:, 2]

            # Update particle positions
            scatter._offsets3d = (x_vals, y_vals, z_vals)

            # Update global best position
            global_best_pos = self.swarm.global_best_history[frame]
            global_best_star._offsets3d = (
                [global_best_pos[0]], [global_best_pos[1]], [global_best_pos[2]]
            )

            global_best_fitness = self.swarm.global_best_fitness
            self.swarm.logger.info(f"Iteration: {frame}, Global Best Fitness = {global_best_fitness}")

            return scatter, global_best_star

        # Create animation
        _ani = FuncAnimation(
            fig,
            update,
            frames=len(self.swarm.history),
            interval=50,
            blit=False
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Fitness Landscape with Particles")
        ax.legend()

        plt.show()
