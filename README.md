# Assignment 1 - Computing the area of the Mandelbrot Set
In this repository we explore the properties of the famous Mandelbrot set, with the specific objective of estimating the set measure by integrating its area with sampling techniques. In particular, we show how Monte Carlo integration can be used to approximate the measure of the set, and how the approximate measure converges to the true measure by refining the number of samples and the number of iterations for the Mandelbrot function. We then show how varying the integration method with variance reduction techniques refines the standard Monte Carlo integration method, leading to a faster convergence of the measure estimation to the true value.

## Structure of the repository
```
root/
├── images/                                    # visulization results
├── src/
│   ├── main.py                                # Main Python script for executing the sampling
│   ├── mandelbrot.py                          # Class implementation for Mandelbrot analysis
│   ├── mt19937.py                             # Class implementation for MT19937 algorithm
│   ├── sampling.py                            # Class implementation for sampling functions
│   └── utilities.py                           # Class implementation for utilities functions
├── ortho-pack/                                # Library for orthogonal sampling
├── README.md
├── Assignment 1 - MANDELBROT.pdf              # Assignment descripition
├── LICENCE.txt                                # Licence file
└── TODO.md                                    # ToDo list
```

## Run configuration
Move inside the src folder and run the main script via `python main.py`