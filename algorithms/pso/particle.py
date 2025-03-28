from typing import Callable
import numpy as np

class Particle:
    def __init__(self, dimension: int, lower_bound: float, upper_bound: float, fitness_func: Callable[[np.ndarray], float]):
        """
        Initialize particle with random position and velocity.

        Args:
            dimension (int): Number of search space dimensions.
            lower_bound (float): Lower bound for position.
            upper_bound (float): Upper bound for position.
        """
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        pmin, pmax, vmin, vmax = self.get_spacial_bounds(fitness_func=fitness_func)
        # Position initialized to a random value between
        # lower_bound and upper_bound for each dimension
        self.position = np.random.uniform(pmin, pmax, dimension)
        self.velocity = np.random.uniform(vmin, vmax, dimension)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def evaluate_fitness(self, fitness_func: Callable[[np.ndarray], float]):
        """
        Evaluate fitness and update the best fitness/position if needed.

        Args:
            fitness_func (Callable): Function to evaluate particle fitness.
        """
        current_fitness = fitness_func(self.position)
        
        # If current fitness is better than the best,
        # update the best fitness and best position
        if current_fitness < self.best_fitness:
            self.best_fitness = current_fitness
            self.best_position = self.position.copy()

    def update_velocity(
        self, 
        global_best_position: np.ndarray, 
        inertia: float=0.7298, 
        cognitive_weight: float=2.05, 
        social_weight: float=2.05,
        ):
        """
        Update velocity based on inertia, cognitive, and social components.

        Args:
            global_best_position (np.array): Best position found by the swarm.
            inertia (float): Weight on previous velocity (default 0.7298).
            cognitive_weight (float): Weight on personal exploration (default 2.05).
            social_weight (float): Weight on swarm exploration (default 2.05).
        """
        prev_velocity = self.velocity.copy()
        # Generate random weights for personal and global influence.
        cognitive_rand = np.random.uniform(0, cognitive_weight, self.dimension)
        social_rand = np.random.uniform(0, social_weight, self.dimension)
        # Calculate new velocity based on PSO velocity formula
        new_velocity = inertia * (
            self.velocity + 
            cognitive_rand * (self.best_position - self.position) + 
            social_rand * (global_best_position - self.position)
        )

        self.velocity = new_velocity
        # print(f"Previous Velocity: {prev_velocity}")
        # print(f"Updated Velocity: {self.velocity}")

    def update_position(self):
        """
        Update the particle's position by adding new velocity.

        Ensure position stays within defined bounds.
        """
        self.position += self.velocity
        # Ensure position stays within bounds
        #self.position = np.clip(self.position, self.lower_bound, self.upper_bound)

    def get_spacial_bounds(self, fitness_func: Callable[[np.ndarray], float]):
        if fitness_func.__name__ == 'square':
            pmin, pmax, vmin, vmax = 0.5, 10.0, 0.001, 0.005
        elif fitness_func.__name__ == 'rosenbrock':
            pmin, pmax, vmin, vmax = 15.0, 30.0, -2.0, 2.0
        elif fitness_func.__name__ == 'ackley':
            pmin, pmax, vmin, vmax = 16.0, 32.0, -2.0, 4.0
        elif fitness_func.__name__ == 'sphere':
            pmin, pmax, vmin, vmax = 15.56, 25.12, -2.0, 4.0
        elif fitness_func.__name__ == 'rastrigin':
            pmin, pmax, vmin, vmax = 2.56, 5.12, -5.0, 4.0
        else:
            # Default case: Raise an error or return default bounds
            print(f"Unknown fitness function: {fitness_func.__name__}")
            raise ValueError(f"Unknown fitness function: {fitness_func.__name__}")

        return pmin, pmax, vmin, vmax