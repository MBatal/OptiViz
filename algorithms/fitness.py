from typing import Callable

import numpy as np


def get_spacial_bounds(fitness_func: Callable[[np.ndarray], float]):
        """
        Returns lower and upper bounds of initial particle position and velocity
        Position values also used to set bounds of plot

        Args:
            fitness_func (Callable): The function PSO is optimizing

        Returns:
        tuple: A tuple containing four values:
            - pmin (float): Lower bound for particle position
            - pmax (float): Upper bound for particle position
            - vmin (float): Lower bound for particle velocity
            - vmax (float): Upper bound for particle velocity
        """
        if fitness_func.__name__ == 'square':
            # pmin, pmax, vmin, vmax = 0.5, 10.0, 0.001, 0.005
            pmin, pmax, vmin, vmax = -0.5, 0.5, 0.001, 0.005
        elif fitness_func.__name__ == 'rosenbrock':
            # pmin, pmax, vmin, vmax = 15.0, 30.0, -2.0, 2.0
            pmin, pmax, vmin, vmax = -5, 5.0, -2.0, 2.0
        elif fitness_func.__name__ == 'ackley':
            # pmin, pmax, vmin, vmax = 16.0, 32.0, -2.0, 4.0
            pmin, pmax, vmin, vmax = -5, 6, -2.0, 4.0
        elif fitness_func.__name__ == 'sphere':
            pmin, pmax, vmin, vmax = -15.56, 15.56, -2.0, 4.0
        elif fitness_func.__name__ == 'rastrigin':
            # pmin, pmax, vmin, vmax = -2.56, 5.12, -5.0, 4.0
            pmin, pmax, vmin, vmax = -2.06, 2.06, -5.0, 4.0
        else:
            print(f"Unknown fitness function: {fitness_func.__name__}")
            raise ValueError(f"Unknown fitness function: {fitness_func.__name__}")

        return pmin, pmax, vmin, vmax

def square(position: np.ndarray) -> float:
    """
    Simple fitness function that calculates the
    sum of squares of the particle's position

    Args:
        position (np.ndarray): The position of the particle in n-dimensional space

    Returns:
        float: The fitness value of the particle,
        which is the sum of the squares of the position
    """
    return np.sum(position**2)

def rosenbrock(position: np.ndarray, a: float=1, b: float=100) -> float:
    """
    Rosenbrock function

    Args:
        position (np.ndarray): The position of the particle in n-dimensional space
        b (float): alpha - sets minimum of function (default 1)
        b (float): beta - sets steepness of curve (default 100)

    Returns:
        float: The fitness value of the particle
    """
    n = len(position)
    return np.sum([(a - position[i])**2 + b * (position[i+1] - position[i]**2)**2 for i in range(n-1)])

def ackley(position: np.ndarray) -> float:
    """
    Ackley function

    Args:
        position (np.ndarray): The position of the particle in n-dimensional space

    Returns:
        float: The fitness value of the particle
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(position)

    return -a * np.exp(-b * np.sqrt(np.sum(position**2) / d)) - np.exp(np.sum(np.cos(c * position)) / d) + a + np.e

def sphere(position: np.ndarray) -> float:
    """
    Sphere function

    Args:
        position (np.ndarray): The position of the particle in n-dimensional space

    Returns:
        float: The fitness value of the particle
    """
    return np.sum(position**2)


def rastrigin(position: np.ndarray) -> float:
    """
    Rastrigin function

    Args:
        position (np.ndarray): The position of the particle in n-dimensional space

    Returns:
        float: The fitness value of the particle
    """
    A = 10
    return A * len(position) + np.sum(position**2 - A * np.cos(2 * np.pi * position))
