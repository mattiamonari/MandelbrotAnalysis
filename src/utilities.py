import numpy as np
import matplotlib.pyplot as plt

from mandelbrot import mandelbrot_area

class Utilities():
    def run_multple_simulations(sampler, samples_range, iters_range, runs, xy_range, 
                                plot = False):
        # store computed areas, and confints
        # increasing iters on rows, increasing samples on cols
        areas = np.empty((len(iters_range), len(samples_range)))
        conf_int = np.empty((len(iters_range), len(samples_range)))

        for j, samples in enumerate(samples_range):
            for i, iters in enumerate(iters_range):
                print(f"Running {sampler.__name__} with {samples} samples and {iters} iterations...")
                res = np.zeros(runs, dtype=float)
                for l in range(runs):
                    points_evaluated = mandelbrot_area(iters, sampler, 
                                                    (samples, xy_range), None)
                    res[l] = (len(points_evaluated[points_evaluated != 0]) / 
                            samples * (xy_range[1] - xy_range[0])**2)

                X = sum(res) / runs
                res -= X
                S2 = (sum(res ** 2) / (runs - 1))
                conf_inter = 1.96 * (S2 / runs)**0.5

                print(f'--> Sampler: {sampler.__name__}')
                print(f'--> Samples: {samples}')
                print(f'--> Iterations: {iters}')
                print(f'--> Est. Mean Area: {X} +/- {conf_inter}')
                print(f'--> Est. Variance: {S2}')

                conf_int[i, j] = conf_inter
                areas[i, j] = X 
        

        plot and plot_multiple_simulations(iters_range, samples_range, areas, 
                                        conf_int, sampler.__name__)
        
        return areas, conf_int


def plot_multiple_simulations(iters_range, samples_range, areas, conf_int, 
                              samp_func_name):
    colors = ['gold', 'limegreen', 'steelblue']
    for i, iter in enumerate(iters_range):
        print(f'Areas: {areas[i]}')
        # Plot the main line with fewer markers
        plt.semilogx(samples_range, areas[i], label=f'Iterations: {iter}', 
                    color=colors[i], marker='o', markersize=4)
    
        # Add confidence interval as a shaded region
        lb = areas[i] - conf_int[i]
        ub = areas[i] + conf_int[i]
        plt.fill_between(samples_range, lb, ub, alpha=0.2, color=colors[i])

    # Add a horizontal line for the true value
    true_area = 1.506484531122182
    plt.axhline(y=true_area, color='black', linestyle='--', label='True Area')

    # # Add text label for true area
    plt.text(samples_range[0] - 20, true_area, f'{true_area:.3f}',
        verticalalignment='bottom', color='black')

    plt.xlabel('Number of Samples')
    plt.ylabel('Estimated Area')
    plt.title('Mandelbrot Set Area Estimation with 95% Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'../images/{samp_func_name}.pdf')
    plt.close()