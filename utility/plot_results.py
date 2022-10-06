# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt


def plot_comparison_hist(values, labels, colors, x_label, y_label):
    for i, item in enumerate(values):
        pyplot.hist(item, color=colors[i], bins=100, alpha=0.5, label=labels[i])
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.legend(loc='upper left')
    pyplot.savefig(f"./results/hist_profit.png")
    pyplot.close()

def plot_comparison_bar(values, labels, x_label, y_label, title):
    plt.rc("axes", axisbelow = True)
    for i, item in enumerate(values):
        pyplot.bar(labels, values, 0.5)
    pyplot.grid()
    pyplot.xticks([i+1 for i in range(len(values))])
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.title(title)
    pyplot.savefig(f"./results/"+str(title)+".png")
    pyplot.close()
