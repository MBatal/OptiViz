import numpy as np

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

