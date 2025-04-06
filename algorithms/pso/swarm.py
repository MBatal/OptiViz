import numpy as np
from algorithms.pso.particle import Particle
from typing import Callable, List
import math
from algorithms.fitness import square, ackley
import logging

class Swarm:
    def __init__(
        self, 
        dimension: int = 3,  
        n_particles: int = 30, 
        lower_bound: float = 0.0, 
        upper_bound: float = 1.0, 
        inertia: float = 0.7298, 
        cognitive_weight: float = 2.05, 
        social_weight: float = 2.05,
        fitness_func: Callable = square,
        n_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
    ):
        """
        Initialize a swarm of particles.

        Args:
            dimension (int): Number of search space dimensions (default 3).
            n_particles (int): Number of particles in the swarm (default 30).
            lower_bound (float): Minimum position value (default 0.0).
            upper_bound (float): Maximum position value (default 1.0).
            inertia (float): Influence of previous velocity (default 0.7298).
            cognitive_weight (float): Influence of personal best (default 2.05).
            social_weight (float): Influence of global best (default 2.05).
            n_iterations (int): Max number of iterations to run (default 1000).
            convergence_threshold (float): Tolerance for convergence (default 1e-6).
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

        self.dimension=dimension
        self.n_particles = n_particles
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.fitness_func = fitness_func
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        
        self.global_best_position = np.random.uniform(low=lower_bound,
                                                      high=upper_bound,
                                                      size=dimension
                                                    )
        self.global_best_fitness = math.inf
        
        # Initialize particles
        self.particles: List[Particle] = [
            Particle(dimension, lower_bound, upper_bound, fitness_func=fitness_func) for _ in range(n_particles)
        ]

        #self.visualizer = SwarmVisualizer(self, fitness_func=fitness_func)

    def evaluate_particles(self):
        """
        Evaluate fitness of all particles

        """
        for particle in self.particles:
            particle.evaluate_fitness(fitness_func=self.fitness_func)
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = particle.position.copy()

    def update_velocity(self):
        """
        Update velocity of all particles.

        """
        for particle in self.particles:
            particle.update_velocity(
                global_best_position=self.global_best_position, 
                inertia=self.inertia, 
                cognitive_weight=self.cognitive_weight, 
                social_weight=self.social_weight
            )

    def update_position(self):
        """
        Update position of all particles.

        """
        for particle in self.particles:
            particle.update_position()

    def optimize(self):
        """
        Run PSO optimization loop

        Returns:
            Tuple[np.ndarray, float]: The best position found and its fitness value.

        """
        for iter in range(self.n_iterations):
            self.evaluate_particles()
            if self.global_best_fitness <= self.convergence_threshold:
                break

            self.update_velocity()
            self.update_position()
            
            # if iter % 10 == 0:
            #     self.visualizer.update()
            if iter % 10 ==0:
                self.logger.info(f'Iteration:={iter}, Best Fitness={self.global_best_fitness}')

        
        # for i, particle in enumerate(self.particles):
        #     logging.info(f"Particle {i} Position: {particle.position}")
        logging.info(f"Iteration: {iter}")
        return self.global_best_position, self.global_best_fitness
    
if __name__ == "__main__":
    swarm = Swarm(dimension=30, n_particles=100, fitness_func=ackley)
    best_position, best_fitness = swarm.optimize()
    print(f"Best Position: {best_position}, Best Fitness: {best_fitness}")