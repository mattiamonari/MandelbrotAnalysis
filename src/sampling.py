import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc
import math

from mandelbrot import is_point_in_mandelbrot, mandelbrot_area
from mt19937 import MT19937, RandomSupport

__all__ = ["random_sampling", "latin_hypercube_sampling", "orthogonal_sampling", 
           "run_multple_simulations"]

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

    print(x_grid)

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

def run_multple_simulations(sampler, samples_range, iters_range, runs, xy_range, plot = False):
    # store computed areas, and confints
    # increasing iters on rows, increasing samples on cols
    areas = np.empty((len(iters_range), len(samples_range)))
    conf_int = np.empty((len(iters_range), len(samples_range)))


    for j, samples in enumerate(samples_range):
        for i, iters in enumerate(iters_range):
            res = np.zeros(runs, dtype=float)
            for l in range(runs):
                points_samples = sampler(samples, xy_range)
                points_evaluated = mandelbrot_area(samples, iters, points_samples, False)
                res[l] = len(points_evaluated[points_evaluated != 0]) / samples * (xy_range[1] - xy_range[0])**2

            X = sum(res) / runs
            res -= X
            S2 = (sum(res ** 2) / (runs - 1))
            conf_inter = 1.96 * (S2 / runs)**0.5
            print(f'Sampler: {sampler.__name__}')
            print(f'Samples: {samples}')
            print(f'Iterations: {iters}')
            print(f'Est. Mean Area: {X} +/- {conf_inter}')
            print(f'Est. Variance: {S2}')
            conf_int[i, j] = conf_inter
            areas[i, j] = X 
    

    colors = ['gold', 'limegreen', 'steelblue']
    for i, iter in enumerate(iters_range):
        print(f'Areas: {areas[i]}')
        # Plot the main line with fewer markers
        plt.semilogx(samples_range, areas[i], 
                    label=f'Iterations: {iter}', 
                    color=colors[i], 
                    marker='o',
                    markersize=4)         # Plot markers every nth point
    
        # Add confidence interval as a shaded region
        plt.fill_between(samples_range,
                        areas[i] - conf_int[i],
                        areas[i] + conf_int[i],
                        alpha=0.2,
                        color=colors[i])

    # Add a horizontal line for the true value
    true_area = 1.506484531122182
    plt.axhline(y=true_area, color='black', linestyle='--', label='True Area')

    # # Add text label for true area
    plt.text(samples_range[0] - 20,
        true_area,
        f'{true_area:.3f}',
        verticalalignment='bottom',
        color='black')

    plt.xlabel('Number of Samples')
    plt.ylabel('Estimated Area')
    plt.title('Mandelbrot Set Area Estimation with 95% Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'../images/{sampler.__name__}.pdf')
    plt.close()