"""
Utilities for Plotting Learning Curves
Written by Patrick Coady (pcoady@alum.mit.edu)
"""
import matplotlib.pyplot as plt
import pylab

def plot_results(results_list):
    # %pylab inline
    pylab.rcParams['figure.figsize'] = (8, 8)

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.set_yscale('log')
    ax0.set_ylim(1, 10)
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('NCE Loss')
    ax0.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.set_ylim(1, 10)
    ax1.set_ylabel('NCE Loss')
    ax1.set_title('Validation Loss')
    markers = ['D', '+', 's', '^', 'v', 'o', '*', '|', '>', 'h']
    for i in range(len(results_list)):
        e_train = results_list[i][1]['e_train']
        e_val = results_list[i][1]['e_val']
        epoch = [i for i in range(1, len(e_train)+1)]
        ax0.plot(epoch, e_train, label=str(i), marker=markers[i], markevery=10)
        ax1.plot(epoch, e_val, label=str(i), marker=markers[i], markevery=10)

    ax0.legend(loc='lower left')
    ax1.legend(loc='lower left')
    plt.tight_layout()
    plt.show()