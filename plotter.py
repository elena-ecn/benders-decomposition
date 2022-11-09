import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_convergence(upper_bounds, lower_bounds):
    """Plots the convergence of the algorithm (upper & lower bounds vs iteration)."""

    sns.set_theme()
    fig, ax = plt.subplots()
    ax.set_title("Benders Decomposition Convergence")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective function bounds')
    iters = np.arange(1, len(upper_bounds)+1, 1, dtype=int)
    plt.plot(iters, upper_bounds, 'o-', label='Upper bound')
    plt.plot(iters, lower_bounds, 'o-', label='Lower bound')
    plt.xticks(iters)
    plt.legend()
    plt.show()
    plt.savefig('images/convergence.png')
