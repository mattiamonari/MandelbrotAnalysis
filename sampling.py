import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
import math

from mandelbrot import is_point_in_mandelbrot
from mt19937 import MT19937, RandomSupport

__all__ = ["random_sampling", "latin_hypercube_sampling", "orthogonal_sampling", "stratified_sampling", "compare_variances", "importance_sampling_mandelbrot", "compare_all_methods", "plot_sampling_distribution"]

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


def stratified_sampling(n_samples=10000, n_strata=10, iters=100, xy_range=(-2, 2)):
    l, u = xy_range
    strata_size = (u - l) / n_strata
    samples_per_stratum = n_samples // (n_strata * n_strata)
    stratum_areas = np.zeros(iters)

    for k in range(iters):
        total_estimate = 0
        for i in range(n_strata):
            for j in range(n_strata):
                x_min = l + i * strata_size
                x_max = l + (i + 1) * strata_size
                y_min = l + j * strata_size
                y_max = l + (j + 1) * strata_size
                
                # Sample points in this stratum
                x = np.random.uniform(x_min, x_max, samples_per_stratum)
                y = np.random.uniform(y_min, y_max, samples_per_stratum)
                
                # Compute indicators for this stratum
                stratum_indicators = np.array([is_point_in_mandelbrot(x[m], y[m], 1000) 
                                            for m in range(samples_per_stratum)])
                
                # Compute proportion in this stratum
                stratum_p = stratum_indicators.mean()
                
                # Area of this stratum
                stratum_area = strata_size * strata_size
                
                # Add this stratum's contribution to total
                total_estimate += stratum_p * stratum_area
                
        stratum_areas[k] = total_estimate
    
    strat_variance = np.var(stratum_areas)

    return stratum_areas.mean(), strat_variance


def compare_variances(n_samples=10000, n_strata=10, iters=100, xy_range=(-2, 2)):
    # Regular Monte Carlo
    l, u = xy_range
    total_area = (u - l)**2
    mc_areas = np.zeros(iters)
    
    for j in range(iters):
        x = np.random.uniform(l, u, n_samples)
        y = np.random.uniform(l, u, n_samples)
        indicators = np.array([is_point_in_mandelbrot(x[i], y[i], 1000) 
                             for i in range(n_samples)])
        
        mc_p = indicators.mean()
        mc_areas[j] = mc_p * total_area
    
    mc_variance = np.var(mc_areas)
    
    # Stratified sampling
    strata_size = (u - l) / n_strata
    samples_per_stratum = n_samples // (n_strata * n_strata)
    stratum_areas = np.zeros(iters)
    
    for k in range(iters):
        total_estimate = 0
        for i in range(n_strata):
            for j in range(n_strata):
                x_min = l + i * strata_size
                x_max = l + (i + 1) * strata_size
                y_min = l + j * strata_size
                y_max = l + (j + 1) * strata_size
                
                # Sample points in this stratum
                x = np.random.uniform(x_min, x_max, samples_per_stratum)
                y = np.random.uniform(y_min, y_max, samples_per_stratum)
                
                # Compute indicators for this stratum
                stratum_indicators = np.array([is_point_in_mandelbrot(x[m], y[m], 1000) 
                                            for m in range(samples_per_stratum)])
                
                # Compute proportion in this stratum
                stratum_p = stratum_indicators.mean()
                
                # Area of this stratum
                stratum_area = strata_size * strata_size
                
                # Add this stratum's contribution to total
                total_estimate += stratum_p * stratum_area
                
        stratum_areas[k] = total_estimate
    
    strat_variance = np.var(stratum_areas)
    
    return (mc_areas.mean(), mc_variance, 
            stratum_areas.mean(), strat_variance)


def importance_sampling_mandelbrot(n_samples=10000, iters=100, xy_range=(-2, 2)):
    """
    Compute Mandelbrot set area using importance sampling with normal distribution
    """
    l, u = xy_range
    total_area = (u - l)**2
    
    # Parameters for importance sampling distribution
    # We'll use a normal distribution centered at -0.5 (main bulb of Mandelbrot)
    # with standard deviation 0.8 to cover most of the interesting region
    mu_x, sigma_x = -0.45, 0.5
    mu_y, sigma_y = 0, 0.5
    
    is_areas = np.zeros(iters)
    
    for j in range(iters):
        # Generate samples from normal distribution
        
        x_vect = np.random.normal(mu_x, sigma_x, n_samples)
        y_vect = np.random.normal(mu_y, sigma_y, n_samples)
        
        # Calculate weights (ratio of target/proposal distributions)
        # Target is uniform over [l,u]×[l,u], so its density is 1/total_area
        # Proposal is normal, we need its density
        weights = (1 / total_area) / (
            norm.pdf(x_vect, mu_x, sigma_x) * norm.pdf(y_vect, mu_y, sigma_y))

        # Compute indicators
        indicators = np.array([is_point_in_mandelbrot(x_vect[i], y_vect[i], 1000) 
                             for i in range(n_samples)])

        # Compute weighted average
        is_areas[j] = np.mean(indicators * weights) * total_area
    
    is_variance = np.var(is_areas)
    
    return is_areas.mean(), is_variance

def compare_all_methods(n_samples=10000, n_strata=10, iters=100, xy_range=(-2, 2)):
    """
    Compare all three methods: Monte Carlo, Stratified, and Importance Sampling
    """
    # Get results from previous methods
    mc_mean, mc_var, strat_mean, strat_var = compare_variances(
        n_samples, n_strata, iters, xy_range
    )
    
    # Get importance sampling results
    is_mean, is_var = importance_sampling_mandelbrot(
        n_samples, iters, xy_range
    )
    
    print(f"Monte Carlo Results:")
    print(f"  Mean area: {mc_mean:.6f}")
    print(f"  Standard deviation: {np.sqrt(mc_var):.6f}")
    
    print(f"\nStratified Sampling Results:")
    print(f"  Mean area: {strat_mean:.6f}")
    print(f"  Standard deviation: {np.sqrt(strat_var):.6f}")
    print(f"  Variance reduction vs MC: {mc_var/strat_var:.2f}x")
    
    print(f"\nImportance Sampling Results:")
    print(f"  Mean area: {is_mean:.6f}")
    print(f"  Standard deviation: {np.sqrt(is_var):.6f}")
    print(f"  Variance reduction vs MC: {mc_var/is_var:.2f}x")
    
    return mc_mean, mc_var, strat_mean, strat_var, is_mean, is_var

# Helper function to visualize the sampling distribution
def plot_sampling_distribution(n_samples, n_points=1000):
    import matplotlib.pyplot as plt
    
    # Generate points from importance sampling distribution
    mu_x, sigma_x = -0.4, 0.6
    mu_y, sigma_y = 0, 0.6
    
    x_vect = np.random.normal(mu_x, sigma_x, n_samples)
    y_vect = np.random.normal(mu_y, sigma_y, n_samples)

    # Plot points
    plt.figure(figsize=(10, 10))
    plt.scatter(x_vect, y_vect)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('Importance Sampling Distribution')
    plt.grid(True)
    plt.savefig('../images/importance_sampling.pdf')