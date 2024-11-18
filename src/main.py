import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

from sampling import *
from mandelbrot import mandelbrot_area, is_point_in_mandelbrot
from utilities import Utilities

def main(n_samples: int, sampling_range: Tuple, n_samples_range: list, 
         iters_range: list, runs: int, xy_range: Tuple) -> int:
    
    #First analysis
    params = (n_samples, sampling_range)

    mandelbrot_area(iters=100, sample_func=random_sampling, params=params, 
                    filename="random", plot=True, verbose=False)
    mandelbrot_area(iters=100, sample_func=latin_hypercube_sampling, params=params, 
                    filename="latin_hypercube", plot=True, verbose=False)
    mandelbrot_area(iters=100, sample_func=orthogonal_sampling, params=params, 
                    filename="orthogonal", plot=True, verbose=False)

    #Analysis on multiple simulations
    Utilities.run_multiple_simulations(sampler=random_sampling, 
                            samples_range=n_samples_range, 
                            iters_range=iters_range, 
                            runs=runs, 
                            xy_range=xy_range, 
                            plot=True)
    Utilities.run_multiple_simulations(sampler=latin_hypercube_sampling, 
                            samples_range=n_samples_range, 
                            iters_range=iters_range, 
                            runs=runs, 
                            xy_range=xy_range, 
                            plot=True)
    Utilities.run_multiple_simulations(sampler=orthogonal_sampling, 
                            samples_range=n_samples_range, 
                            iters_range=iters_range, 
                            runs=runs, 
                            xy_range=xy_range, 
                            plot=True)
    

    Utilities.run_convergence_analysis(n_runs=runs)

    plot_sampling_distribution(n_samples)


    # Compare all three methods
    results = []

    for i, n in enumerate(n_sampling_range):
        results.append(compare_all_methods(n_samples=n, n_strata=100, iters=10))

    mc_mean = np.array([result[0] for result in results])
    strat_mean = np.array([result[2] for result in results])
    is_mean = np.array([result[4] for result in results])

    mc_var = np.array([result[1] for result in results])
    strat_var = np.array([result[3] for result in results])
    is_var = np.array([result[5] for result in results])


    mc_confint = 1.96 * mc_var**0.5 / np.sqrt(10)
    strat_confint = 1.96 * strat_var**0.5 / np.sqrt(10)
    is_confint = 1.96 * is_var**0.5 / np.sqrt(10)

    Utilities.plot_multiple_simulations([1000, 1000, 1000], n_sampling_range, [mc_mean, strat_mean, is_mean], 
                          [mc_confint, strat_confint, is_confint], "comparison")

    plt.plot(n_sampling_range, mc_var, 'b', label="Monte Carlo")
    plt.plot(n_sampling_range, strat_var, 'r', label="Stratified")
    plt.plot(n_sampling_range, is_var, 'g', label="Importance Sampling")
    plt.xscale('log')
    plt.xlabel("Number of Samples")
    plt.ylabel("Variance")
    plt.title("Variance Reduction with different sampling methods")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../images/variance_reduction_multiple.pdf')


    Utilities.run_timings_test(n_samples, iters_range[-1], xy_range)

    return 0


if __name__ == "__main__":
    # Parameters
    n_sampling_range = np.logspace(2, 6, 20, dtype=int)
    runs = 50
    iters_range = [10, 100, 1000]
    xy_range = (-2, 0.7)

    main(10000, (-2, 2), n_sampling_range, iters_range, runs, xy_range)
