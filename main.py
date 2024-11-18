import time
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

from sampling import *
from mandelbrot import mandelbrot_area, is_point_in_mandelbrot
from utilities import Utilities

def main(n_samples: int, sampling_range: Tuple, n_samples_range: list, 
         iters_range: list, runs: int, xy_range: Tuple) -> int:
    
    # First analysis
    params = (n_samples, sampling_range)

    mandelbrot_area(iters=100, sample_func=random_sampling, params=params, 
                    filename="random", plot=True, verbose=True)
    mandelbrot_area(iters=100, sample_func=latin_hypercube_sampling, params=params, 
                    filename="latin_hypercube", plot=True, verbose=True)
    mandelbrot_area(iters=100, sample_func=orthogonal_sampling, params=params, 
                    filename="orthogonal", plot=True, verbose=True)

    Analysis on multiple simulations
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
    

    fig, results, means = run_convergence_analysis(n_runs=50)
    plt.show()

    # Print some statistics
    for i, iters in enumerate([10, 100, 1000]):
        print(f"\nStatistics for {iters} iterations:")
        print(f"Best relative error: {np.min(np.abs(means[i] - 1.506484) / 1.506484):.6f}")
        print(f"Worst relative error: {np.max(np.abs(means[i] - 1.506484) / 1.506484):.6f}")

    # Compare variances
    n_sampling_range = np.logspace(2, 4, 10, dtype=int)
    mc_var, strat_var = np.zeros(len(n_sampling_range)), np.zeros(len(n_sampling_range))
    mc_area, strat_area = np.zeros(len(n_sampling_range)), np.zeros(len(n_sampling_range))

    for i, n in enumerate(n_sampling_range):
        mc_area[i], mc_var[i], strat_area[i], strat_var[i] = compare_variances(n, 10, 100, (-2, 2))
        print(f"n = {n} -> Monte Carlo variance: {mc_var[i]:.6f}")
        print(f"n = {n} -> Stratified variance: {strat_var[i]:.6f}")
        print(f"n = {n} -> Variance reduction factor: {mc_var[i]/strat_var[i]:.2f}x")

    
    mc_confint = 1.96 * mc_var**0.5 / np.sqrt(n_sampling_range)
    strat_confint = 1.96 * strat_var**0.5 / np.sqrt(n_sampling_range)

    Utilities.plot_multiple_simulations([100], n_sampling_range, [mc_area], [mc_confint], 
                            "monte_carlo")
    Utilities.plot_multiple_simulations([100], n_sampling_range, [strat_area], [strat_confint], 
                            "stratified")
    
    plt.figure(figsize=(8, 8))
    plt.plot(mc_area - mc_confint, 'b', label="Lower Bound (Monte Carlo)")
    plt.plot(mc_area + mc_confint, 'c',label="Upper Bound (Monte Carlo)")
    plt.plot(strat_area - strat_confint, 'r', label="Lower Bound (Stratified Sampling)")
    plt.plot(strat_area + strat_confint, 'orange', label="Upper Bound (Stratified Sampling)")
    
    plt.xlabel("Number of Samples")
    plt.ylabel("Estimated Area")
    plt.title("95% Confidence Intervals for Area Estimation Methods")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../images/confidence_intervals.pdf')

    plt.figure(figsize=(8, 8))
    plt.plot(n_sampling_range, mc_var, 'b', label="Monte Carlo")
    plt.plot(n_sampling_range, strat_var, 'r', label="Stratified")
    plt.xscale('log')
    plt.xlabel("Number of Samples")
    plt.ylabel("Variance")
    plt.title("Variance Reduction with Stratified Sampling")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../images/variance_reduction.pdf')

    iters = 1000
    n_samples = 10000

    l, u = (-2, 2)
    total_area = (u - l)**2
    mc_areas = np.zeros(iters)

    for j in range(iters):
        x = np.random.uniform(l, u, n_samples)
        y = np.random.uniform(l, u, n_samples)
        indicators = np.array([is_point_in_mandelbrot(x[i], y[i], 100) 
                                for i in range(n_samples)])
        
        mc_p = indicators.mean()
        mc_areas[j] = mc_p * total_area

    mc_variance = np.var(mc_areas)
    theoretical_variance = mc_p * (1 - mc_p) / n_samples

    print(f"Estimated Variance: {mc_variance:.6f}")
    print(f"Theoretical Variance: {theoretical_variance:.6f}")


    importance_sampling_mandelbrot(n_samples=100000, iters=10, xy_range=(-2, 2))

    plot_sampling_distribution(n_samples)


    # Compare all three methods
    results = []

    n_sampling_range = [10000, 50000, 100000]

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


    strata_sizes = np.linspace(10, 100, 10, dtype=int)
    timings = np.zeros(10)

    for i, n in enumerate(strata_sizes):
        print(f"Strata Size: {n}")

        start_time = time.time()
        res = stratified_sampling(n_samples=10000, n_strata=n, iters=100, xy_range=(-2, 2))
        timings[i] = time.time() - start_time

    time_start = time.time()
    res = importance_sampling_mandelbrot(n_samples=10000, iters=100, xy_range=(-2, 2))
    time_end = time.time()

    plt.plot(strata_sizes, timings, 'r', label='Stratified Sampling Time')
    plt.plot(strata_sizes, np.ones(10) * importance_sampling_time, 'y', label='Importance Sampling Time')

    plt.xlabel("Number of Strata")
    plt.ylabel("Time (s)")
    plt.title("Computation Time vs. Number of Strata")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../images/time_test.pdf')

    return 0


if __name__ == "__main__":
    # Parameters
    n_sampling_range = np.logspace(2, 6, 20, dtype=int)
    iters = 10
    iters_range = [10, 100, 1000]
    xy_range = (-2, 0.7)

    main(10000, (-2, 2), n_sampling_range, iters_range, iters, xy_range)



    
