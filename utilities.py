import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from collections.abc import Callable

from mandelbrot import mandelbrot_area

class Utilities():
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

    def run_multiple_simulations(sampler: Callable, samples_range: Tuple, 
                                iters_range: list, runs: int, xy_range: Tuple, 
                                plot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # store computed areas, and confints
        # increasing iters along rows, increasing samples along cols
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
    
    def run_convergence_analysis(n_runs=50):
        # Known area of the Mandelbrot set (approximately 1.506484...)
        TRUE_AREA = 1.506484
        
        # Parameters
        iterations = [10, 100, 1000]
        samples = [1000, 3300, 6600, 10000, 33000, 66000, 100000, 330000, 660000, 1000000]
        xy_range = [-2, 2]
        total_area = (xy_range[1] - xy_range[0]) * (xy_range[1] - xy_range[0])
        
        # Store results for each run
        results = np.zeros((len(iterations), len(samples), n_runs))
        rel_errors = np.zeros((len(iterations), len(samples), n_runs))
        
        # Run multiple simulations
        for run in tqdm(range(n_runs), desc="Simulation runs"):
            for i, iters in enumerate(iterations):
                for j, samps in enumerate(samples):
                    points = mandelbrot_area(
                        iters,
                        random_sampling,
                        [samps, [xy_range[0], xy_range[1]]],
                        "convergence",
                        plot=False,
                        verbose=False
                    )
                    # Calculate area
                    area = np.count_nonzero(points) / samps * total_area
                    results[i, j, run] = area
                    rel_errors[i, j, run] = np.abs(area - TRUE_AREA) / TRUE_AREA
        
        # Calculate statistics
        mean_results = np.mean(results, axis=2)
        mean_rel_errors = np.mean(rel_errors, axis=2)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        markers = ['o', 's', '^']
        
        for i, iters in enumerate(iterations):
            # Calculate relative error from true area
            std_error = np.std(rel_errors[i], axis=1)
            conf_inter = 1.96 * std_error / (n_runs - 1)**0.5

            print(samples, mean_rel_errors[i], conf_inter)

            # Plot with error bars
            plt.errorbar(samples, mean_rel_errors[i], 
                        yerr=conf_inter,
                        label=f'{iters} iterations',
                        color=colors[i],
                        marker=markers[i],
                        markersize=8,
                        capsize=5,
                        alpha=0.7,
                        linewidth=2)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Samples')
        plt.ylabel('Relative Error |A_estimated - A_true|/A_true')
        plt.title('Convergence Analysis of Mandelbrot Set Area Estimation\n'
                f'({n_runs} runs per configuration)')
        
        # Add grid with transparency
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add horizontal line at 1% error
        plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.3, 
                    label='1% error threshold')
        
        plt.legend()
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return plt.gcf(), results, mean_results



