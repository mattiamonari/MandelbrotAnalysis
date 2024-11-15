# TO DO

### 1)

Create main function in main.py

### 2)

Typehinting on all functions in all modules

### 3)

Decide variance reduction technique for [4]
- Importance sampling: x ~ N(mu, sigma) ?
- Others?

### 4)

Define convergence study
- Difference A_i,j - A_true 
- More plots?
- More statistics?

### 5)

Code refactor
- sampling.py:
    - run_multple_simulations()

- mandelbrot.py:
    - mandelbrot_area(_, sampled_points, *) -> mandelbrot_area(_, sample_func, params, *)
