# File: src/visualization/convergence_plot.py

import matplotlib.pyplot as plt

def plot_convergence_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history, marker='o', linestyle='-', color='b')
    plt.title('Convergence History', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return plt.gcf()