import time
import numpy as np


from sampling import random_sampling, latin_hypercube_sampling, orthogonal_sampling, run_multple_simulations
from mandelbrot import plot_mandelbrot


#plot_mandelbrot(1000, 100, (-2, 2), (-2, 2))

samples = 10000

print("Starting random sampling...")
start_time = time.time()
random_sampling(samples, 100, (-2, 2), True)
print(f"Random sampling took {time.time() - start_time} seconds")

print("Starting latin hypercube sampling...")
start_time = time.time()
latin_hypercube_sampling(samples, 100, (-2, 2), True)
print(f"Latin hypercube sampling took {time.time() - start_time} seconds")

print("Starting orthogonal sampling...")
start_time = time.time()
orthogonal_sampling(samples, 100, (-2, 0.7), 10, 6969, True)
print(f"Orthogonal sampling took {time.time() - start_time} seconds")

run_multple_simulations(random_sampling, np.logspace(2, 6, 20, dtype=int), [100], 100, (-2, 0.7), True)

run_multple_simulations(latin_hypercube_sampling, np.logspace(2, 6, 20, dtype=int), [100], 100, (-2, 0.7), True)

run_multple_simulations(orthogonal_sampling, np.logspace(2, 6, 20, dtype=int), [100], 100, (-2, 0.7), True)
