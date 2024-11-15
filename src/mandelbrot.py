import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def mandelbrot(x, y, t):
    '''
    This function is used to make colorful plots of the mandelbrot set returning
    the number of iterations it took to escape
    '''
    z = 0
    c = complex(x, y)
    threshold = 2

    for i in range(t - 1):
        if abs(z) > threshold:
            return i-1
        z = z**2 + c

    return t

def mandelbrot_area(x, y, t):
    '''
    This function is used to make evaluate if a point is in the mandelbrot set
    '''
    z = 0
    c = complex(x, y)
    threshold = 2

    for _ in range(t - 1):
        if abs(z) > threshold:
            return 0
        z = z**2 + c

    return 1

def plot_mandelbrot(samples, iters, x_range, y_range):
    cmaps = ["cubehelix", "rocket", "mako", "magma"]
    _, ax = plt.subplots(2, 2, figsize=(10, 10))

    for col in range(len(cmaps)):
        x = np.linspace(x_range[0], x_range[1], samples)
        y = np.linspace(y_range[0], y_range[1], samples)
        z = np.zeros((samples, samples))

        for i in range(samples):
            for j in range(samples):
                z[i, j] = mandelbrot(x[i], y[j], iters)

        a1, a2 = col % 2, col // 2
        
    plt.figure(figsize=(8, 8))
    sns.heatmap(z, cmap=cmaps[col], ax=ax[a1, a2], xticklabels=False, yticklabels=False, cbar=False)
    plt.savefig("../images/mandelbrot.pdf")
