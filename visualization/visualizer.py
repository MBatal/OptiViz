import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable
from algorithms.fitness import square

class SwarmVisualizer:
    def __init__(self, swarm, fitness_func: Callable = square):
        """
        Plot solutions using matplotlib

        Args:
            swarm: Swarm object of solutions.
            fitness_function: Fitness function to evaluate solutions (default: squared).
        """
        self.swarm = swarm
        self.fitness_func = fitness_func

    def plot_function(self):
        """
        Plot the function that is being optimized.
        For 2D and 3D, we plot the fitness landscape.
        """
        x_vals = np.linspace(self.swarm.lower_bound, self.swarm.upper_bound, 100)
        y_vals = np.linspace(self.swarm.lower_bound, self.swarm.upper_bound, 100)
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = self.fitness_func(np.vstack([X.ravel(), Y.ravel()]).T)  # Flatten and pass to the function
        
        Z = Z.reshape(X.shape)
        
        # Create 2D or 3D plot depending on the problem
        fig = plt.figure(figsize=(10, 6))
        
        if self.swarm.dimension == 2:
            ax = fig.add_subplot(111)
            ax.contour(X, Y, Z, levels=20, cmap="viridis")
            ax.scatter(self.swarm.global_best_position[0], self.swarm.global_best_position[1], color="red", label="Best Position")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Function Landscape with Particle Positions")
            
            for particle in self.swarm.particles:
                ax.scatter(particle.position[0], particle.position[1], color="blue", alpha=0.5)
                
        elif self.swarm.dimension == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.6)
            ax.scatter(self.swarm.global_best_position[0], self.swarm.global_best_position[1], self.swarm.global_best_position[2], color="red", label="Best Position")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Function Landscape with Particle Positions")
            
            for particle in self.swarm.particles:
                ax.scatter(particle.position[0], particle.position[1], particle.position[2], color="blue", alpha=0.5)

        plt.show()

    def plot_2d(self):
        """
        Plot 2-Dimensional visualization of solutions.
        """

        fig, ax = plt.subplots()

        # Scatter plot for particles
        x_vals = [p.position[0] for p in self.swarm.particles]
        y_vals = [p.position[1] for p in self.swarm.particles]

        ax.scatter(x_vals, y_vals, c='blue', label="Particles")
        ax.scatter(self.swarm.global_best_position[0], self.swarm.global_best_position[1], c='red', marker='*', s=200, label="Global Best")
        
        ax.set_title("2D Particle Swarm Optimization")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.legend()
        
        plt.show()

    def plot_3d(self):
        """
        Plot 3-Dimensional visualization of solutions.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for particles
        x_vals = [p.position[0] for p in self.swarm.particles]
        y_vals = [p.position[1] for p in self.swarm.particles]
        z_vals = [p.position[2] for p in self.swarm.particles]

        ax.scatter(x_vals, y_vals, z_vals, c='blue', label="Particles")
        ax.scatter(self.swarm.global_best_position[0], self.swarm.global_best_position[1], self.swarm.global_best_position[2], 
                   c='red', marker='*', s=200, label="Global Best")

        ax.set_title("3D Particle Swarm Optimization")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.legend()

        plt.show()

    def plot(self, dim: int):
        """
        Plot the particles based on the dimensionality of the problem.

        Args:
            dim (int): The dimensionality of the problem (2 or 3).
        """
        if dim == 2:
            self.plot_2d()
        elif dim == 3:
            self.plot_3d()
        else:
            print("Visualization is only available for 2D or 3D problems.")

    def update(self):
        """
        Update the visualizer by plotting the particles and their global best.
        """
        # Clear the previous plot (to avoid overlap)
        plt.clf()

        # Plot the particles' positions
        positions = np.array([particle.position for particle in self.swarm.particles])
        plt.scatter(positions[:, 0], positions[:, 1], color='blue', label='Particles')

        # Plot the global best position
        plt.scatter(self.swarm.global_best_position[0], self.swarm.global_best_position[1], color='red', label='Global Best', s=100, marker='X')

        # Plot the fitness function surface
        x = np.linspace(self.swarm.lower_bound, self.swarm.upper_bound, 100)  # Example range for 2D
        y = np.linspace(self.swarm.lower_bound, self.swarm.upper_bound, 100)  # Example range for 2D
        X, Y = np.meshgrid(x, y)

        # Ensure fitness function returns values for the grid of points
        Z = np.array([self.fitness_func(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
        
        # Reshape the result back to the grid shape
        Z = Z.reshape(X.shape)

        # Plot the contour (2D)
        plt.contour(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.5)

        plt.legend()
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Swarm Optimization')
        plt.pause(0.1)  # Pause to allow the plot to update
        plt.show(block=True)  # Non-blocking show

    def animate(self, n_iterations: int, interval: int = 200):
        """
        Create an animation of particles' movement through the search space.

        Args:
            n_iterations (int): Number of iterations for animation.
            interval (int): Time interval between frames in milliseconds.
        """
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots()

        # Set up the plot with the initial position
        x_vals = [p.position[0] for p in self.swarm.particles]
        y_vals = [p.position[1] for p in self.swarm.particles]

        scatter = ax.scatter(x_vals, y_vals, c='blue')

        def update(frame):
            self.swarm.update_velocity()
            self.swarm.update_position()
            # Get updated positions
            x_vals = [p.position[0] for p in self.swarm.particles]
            y_vals = [p.position[1] for p in self.swarm.particles]
            scatter.set_offsets(np.c_[x_vals, y_vals])
            return scatter,

        ani = FuncAnimation(fig, update, frames=n_iterations, interval=interval)
        plt.show()

# Usage example:

# Assuming you have a Swarm object (e.g., swarm = Swarm(...) initialized)
# and you're using the squared fitness function.

# visualizer = SwarmVisualizer(swarm, fitness_function=squared)

# If it's 2D or 3D problem, you can visualize:
# visualizer.plot(dim=2)  # Or dim=3 for 3D