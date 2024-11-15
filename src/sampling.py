import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc
import math

from mandelbrot import mandelbrot_area
from mt19937 import MT19937, RandomSupport

def random_sampling(samples, iters, xy_range, plot = False):
    u1 = np.random.uniform(0, 1, samples)
    u2 = np.random.uniform(0, 1, samples)

    sols = np.zeros((samples), dtype=complex)
    for s in range(samples):
        x = xy_range[0] + (xy_range[1] - xy_range[0]) * u1[s]
        y = xy_range[0] + (xy_range[1] - xy_range[0]) * u2[s]
        iter = mandelbrot_area(x, y, iters)
        sols[s] = complex(x, y) if iter else 0

    if(plot == True):
        plt.figure(figsize=(8, 8))
        plt.grid(True, alpha=0.3)
        plt.scatter(sols.real, sols.imag, color='red', s=0.1)
        plt.savefig("../images/random_sampling.pdf")
    
    return sols

def latin_hypercube_sampling(samples, iters, xy_range, plot = False):
    sampler = qmc.LatinHypercube(d=2)
    lhs_samples = sampler.random(n=samples)

    scaled_samples = qmc.scale(lhs_samples[:], xy_range[0], xy_range[1])
    real_samples, img_samples = scaled_samples[:, 0], scaled_samples[:, 1]

    sols = np.zeros((samples), dtype=complex)

    for s in range(samples):
        iter = mandelbrot_area(real_samples[s], img_samples[s], iters)
        sols[s] = complex(real_samples[s], img_samples[s]) if iter else 0

    if(plot == True):
        plt.figure(figsize=(8, 8))
        plt.grid(True, alpha=0.3)
        plt.scatter(sols.real, sols.imag, color='red', s=0.1)
        plt.savefig("../images/latin_hypercube_sampling.pdf")
    
    return sols

def orthogonal_sampling(samples, iters, xy_range, depth=10, seed = 6969, plot = False):
    size = math.ceil(((samples / 10) ** 0.5))
    rng = MT19937()
    rng.init_genrand(seed)
    rand_support = RandomSupport(rng)
    
    # Initialize lists
    x_grid = [[i + j * size for j in range(size)] for i in range(size)]
    y_grid = [[i + j * size for j in range(size)] for i in range(size)]

    points = []
    scale = (xy_range[1] - xy_range[0]) / (samples/10)
    
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

    print(len(points[points != 0]))

    points = np.array(points[:samples])
    sols = np.zeros((samples), dtype=complex)
    print(samples)
    print(size)
    print(len(points))
    for s in range(samples):
        iter = mandelbrot_area(points[s][0], points[s][1], iters)
        sols[s] = complex(points[s][0], points[s][1]) if iter else 0

    if(plot == True):
        plt.figure(figsize=(8, 8))
        plt.grid(True, alpha=0.3)
        plt.scatter(sols.real, sols.imag, color='red', s=0.1)
        plt.savefig("../images/orthogonal_sampling.pdf")
    
    return sols

def run_multple_simulations(sampler, samples_range, iters_range, runs, xy_range, plot = False):
    for samples in samples_range:
        for iters in iters_range:
            res = np.empty(runs, dtype=float)
            for _ in range(runs):
                sols = sampler(samples, iters, xy_range)
                res = np.append(res, len(sols[sols != 0]) / samples * (xy_range[1] - xy_range[0])**2)
            
            X = sum(res) / runs
            res -= X
            S2 = (sum(res ** 2) / (runs - 1))
            conf_inter = 1.96 * (S2 / runs)**0.5
            print(f'Sampler: {sampler.__name__}')
            print(f'Samples: {samples}')
            print(f'Iterations: {iters}')
            print(f'Est. Mean Area: {X} +/- {conf_inter}')
            print(f'Est. Variance: {S2}')