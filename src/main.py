import time
import numpy as np


from sampling import *
from mandelbrot import mandelbrot_area
from utilities import Utilities

def main(n_samples, sampling_range, n_samples_range, iters_range, runs, xy_range):
    
    # First analysis
    params = (n_samples, sampling_range)

    mandelbrot_area(iters=100, sample_func=random_sampling, params=params, 
                    filename="random", plot=True, verbose=True)
    mandelbrot_area(iters=100, sample_func=latin_hypercube_sampling, params=params, 
                    filename="latin_hypercube", plot=True, verbose=True)
    mandelbrot_area(iters=100, sample_func=orthogonal_sampling, params=params, 
                    filename="orthogonal", plot=True, verbose=True)

    # Analysis on multiple simulations
    Utilities.run_multple_simulations(sampler=random_sampling, 
                            samples_range=n_samples_range, 
                            iters_range=iters_range, 
                            runs=runs, 
                            xy_range=xy_range, 
                            plot=True)
    Utilities.run_multple_simulations(sampler=latin_hypercube_sampling, 
                            samples_range=n_samples_range, 
                            iters_range=iters_range, 
                            runs=runs, 
                            xy_range=xy_range, 
                            plot=True)
    Utilities.run_multple_simulations(sampler=orthogonal_sampling, 
                            samples_range=n_samples_range, 
                            iters_range=iters_range, 
                            runs=runs, 
                            xy_range=xy_range, 
                            plot=True)

    return 0


if __name__ == "__main__":
    # Parameters
    n_sampling_range = np.logspace(2, 6, 20, dtype=int)
    iters = 100
    iters_range = [10, 100, 1000]
    xy_range = (-2, 0.7)

    main(10000, (-2, 2), n_sampling_range, iters_range, iters, xy_range)