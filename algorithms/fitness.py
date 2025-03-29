import numpy as np
import math

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
