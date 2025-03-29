import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable
import logging

class SwarmVisualizer:
    def __init__(self, swarm, fitness_func: Callable, clip: bool=False):
        """
        Initialize the visualizer.

        Args:
            swarm: Swarm object with particle positions and global best.
            fitness_func: Fitness function to evaluate the landscape.
        """
        self.swarm = swarm
        self.fitness_func = fitness_func
        self.clip = clip

    def plot_function(self):
        """
        Plot the fitness landscape (function) and particles.
        """
        if self.swarm.dimension == 2:
            self.plot_2d()
        elif self.swarm.dimension == 3:
            self.plot_3d()
        else:
            print("Visualization is only available for 2D or 3D problems.")

    def plot_2d(self):
        """
        Plot 2D visualization of the function and particles.
        """
        x_vals = np.linspace(self.swarm.lower_bound, self.swarm.upper_bound, 100)
        y_vals = np.linspace(self.swarm.lower_bound, self.swarm.upper_bound, 100)
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array([self.fitness_func(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
        
        Z = Z.reshape(X.shape)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the fitness function as contour
        ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)

        # Plot the particles
        x_vals = [p.position[0] for p in self.swarm.particles]
        y_vals = [p.position[1] for p in self.swarm.particles]
        ax.scatter(x_vals, y_vals, c='blue', label="Particles", alpha=0.7)

        # Plot the global best position
        ax.scatter(self.swarm.global_best_position[0], self.swarm.global_best_position[1],
                   c='red', marker='*', s=200, label="Global Best")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Fitness Landscape with Particles")
        ax.legend()

        plt.show()

    def plot_3d(self):
        """
        Plot 3D visualization of the function and particles.
        """
        x_vals = np.linspace(self.swarm.lower_bound, self.swarm.upper_bound, 100)
        y_vals = np.linspace(self.swarm.lower_bound, self.swarm.upper_bound, 100)
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array([self.fitness_func(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
        Z = Z.reshape(X.shape)
        if self.clip:
            Z = np.clip(Z, 0, np.percentile(Z, 15))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the fitness function as surface
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)

        # Plot the particles
        x_vals = [p.position[0] for p in self.swarm.particles]
        y_vals = [p.position[1] for p in self.swarm.particles]
        z_vals = [p.position[2] for p in self.swarm.particles]
        ax.scatter(x_vals, y_vals, z_vals, c='blue', label="Particles", alpha=0.7)

        # Plot the global best position
        ax.scatter(self.swarm.global_best_position[0], self.swarm.global_best_position[1],
                   self.swarm.global_best_position[2], c='red', marker='*', s=200, label="Global Best")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Fitness Landscape with Particles")
        ax.legend()

        plt.show()

    def animate(self, n_iterations: int, interval: int = 25):
        """
        Animate the PSO process over time and track the global best.
        
        Args:
            n_iterations (int): Number of iterations.
            interval (int): Time interval between frames in milliseconds.
        """
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Initial positions of particles
        x_vals = [p.position[0] for p in self.swarm.particles]
        y_vals = [p.position[1] for p in self.swarm.particles]
        z_vals = [p.position[2] for p in self.swarm.particles]

        # Create the initial scatter plot
        scatter = ax.scatter(x_vals, y_vals, z_vals, c='blue', label="Particles")

        # Create the global best marker
        best_marker = ax.scatter(self.swarm.global_best_position[0], 
                                self.swarm.global_best_position[1], 
                                self.swarm.global_best_position[2], 
                                c='red', marker='*', s=200, label="Global Best")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('PSO Animation')

        ax.legend()

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

        def update(frame):
            # Update velocity and position of particles
            self.swarm.evaluate_particles()
            self.swarm.update_velocity()
            self.swarm.update_position()

            # Get updated particle positions
            x_vals = [p.position[0] for p in self.swarm.particles]
            y_vals = [p.position[1] for p in self.swarm.particles]
            z_vals = [p.position[2] for p in self.swarm.particles]

            # Update particle positions in the scatter plot
            scatter._offsets3d = (x_vals, y_vals, z_vals)

            # Update the global best position marker
            best_marker._offsets3d = ([self.swarm.global_best_position[0]], 
                                    [self.swarm.global_best_position[1]], 
                                    [self.swarm.global_best_position[2]])
            
            x_vals_range = np.linspace(self.swarm.lower_bound, self.swarm.upper_bound, 100)
            y_vals_range = np.linspace(self.swarm.lower_bound, self.swarm.upper_bound, 100)
            X, Y = np.meshgrid(x_vals_range, y_vals_range)

            # Calculate the Z values (fitness function) for each (X, Y)
            Z = np.array([self.swarm.fitness_func(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
            Z = Z.reshape(X.shape)
            
            if self.clip:
                Z = np.clip(Z, 0, np.percentile(Z, 15))

            # Update the fitness surface plot
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)


            # Log the iteration number, global best fitness, and global best position
            global_best_fitness = self.swarm.global_best_fitness
            logging.info(f"Global Best Fitness = {global_best_fitness}, Position = {self.swarm.global_best_position}")

            return scatter, best_marker

        # Create the animation
        ani = FuncAnimation(fig, update, frames=n_iterations, interval=interval)

        # Display the animation
        plt.show()  # Show the animation window, and it will run the updates.

