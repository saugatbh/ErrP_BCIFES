
"""
This script plots the correctness of the online motor-imagery
experiment for both FES and VIS feedback

Usage: python online_correctness.py

Author: Saugat Bhattacharyya,
        Ulster University
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use("paper_style.mplstyle")

classes = np.load('pickles/pre-processed_epochs.npy', allow_pickle=True)[()]['classes']
people = 8
experiment = ['fes', 'vis']
index = np.arange(people)
bar_width = .9
opacity = 1
correctness = np.zeros(people)

for expt in experiment:
    print('Experiment %s' % expt)
    for sub in range(people):
        print('Subject %s:' % str(sub + 1), Counter(classes[expt][sub]))
        correct, incorrect = Counter(classes[expt][sub])[-1], Counter(classes[expt][sub])[1]
        correctness[sub] = (correct * 1.0 / (correct + incorrect)) * 100

    # Plot result
    plt.figure(figsize=(8, 6))
    plt.bar(index, correctness, bar_width, alpha=opacity, color='b',
            label='Decision Accuracy (Mean = %.2f, Std = %.3f)' % (np.mean(correctness), np.std(correctness)))
    plt.xlabel('Participants')
    plt.ylabel('Decision Accuracy (\%)')
    plt.xlim(-.5, 7.5)
    plt.ylim(0, 100)
    if expt == 'vis':
        plt.title('VIS Group')
        plt.xticks(index, ("VIS01", "VIS02", "VIS03", "VIS04", "VIS05", "VIS06", "VIS07", "VIS08"))
    elif expt == 'fes':
        plt.title('FES Group')
        plt.xticks(index, ("FES01", "FES02", "FES03", "FES04", "FES05", "FES06", "FES07", "FES08"))
    plt.legend(loc='upper center')
    plt.savefig("results/fig_decision_accuracy_" + expt + ".pdf")