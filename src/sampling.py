import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc
import math

from mandelbrot import is_point_in_mandelbrot, mandelbrot_area
from mt19937 import MT19937, RandomSupport

__all__ = ["random_sampling", "latin_hypercube_sampling", "orthogonal_sampling"]

def random_sampling(samples, xy_range):
    u1 = np.random.uniform(0, 1, samples)
    u2 = np.random.uniform(0, 1, samples)

    x = xy_range[0] + (xy_range[1] - xy_range[0]) * u1
    y = xy_range[0] + (xy_range[1] - xy_range[0]) * u2

    return x, y

def latin_hypercube_sampling(samples, xy_range):
    sampler = qmc.LatinHypercube(d=2)
    lhs_samples = sampler.random(n=samples)

    scaled_samples = qmc.scale(lhs_samples[:], xy_range[0], xy_range[1])
    real_samples, img_samples = scaled_samples[:, 0], scaled_samples[:, 1]

    return real_samples, img_samples

def orthogonal_sampling(samples, xy_range, depth=10, seed = 6969):
    size = math.ceil(((samples / depth) ** 0.5))
    rng = MT19937()
    rng.init_genrand(seed)
    rand_support = RandomSupport(rng)
    
    # Initialize lists
    x_grid = [[i + j * size for j in range(size)] for i in range(size)]
    y_grid = [[i + j * size for j in range(size)] for i in range(size)]

    points = []
    scale = (xy_range[1] - xy_range[0]) / (samples/depth)
    
    for _ in range(depth):
        # Permute lists
        for i in range(size):
            rand_support.permute(x_grid[i])
            rand_support.permute(y_grid[i])
        
        # Generate points
        for i in range(size):
            for j in range(size):
                x = xy_range[0] + scale * (x_grid[i][j] + rng.genrand_real2())
                y = xy_range[0] + scale * (y_grid[j][i] + rng.genrand_real2())
                points.append((x, y))

    points = np.array(points[:samples])

    return (points[:,0], points[:,1])