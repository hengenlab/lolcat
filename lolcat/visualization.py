import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names=None, xlabel='Predicted label', ylabel='True label'):
    figure = plt.figure(figsize=(8, 8))
    cm_total = np.block([[cm, cm.sum(axis=1)[:, np.newaxis]], [cm.sum(axis=0), 0]])

    cm = cm / cm.sum(axis=1).reshape(-1, 1)

    cm = np.block([[cm, np.zeros((cm.shape[0], 1))], [np.zeros((cm.shape[1],)), 0]])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    if class_names == None:
        class_names = list(range(cm.shape[0] - 1))
    class_names = class_names + ['total']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        is_total_row = i == (cm.shape[0] + 1)
        is_total_col = j == (cm.shape[1] + 1)
        if not (is_total_row and is_total_col):
            color = "white" if cm[i, j] > threshold and not (is_total_row or is_total_col) else "black"
            plt.text(j, i, '%.2f' % cm_total[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return figure


def visualize_sample(x, y, logits, class_names, uncertainty=None):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={'width_ratios': [2, 1]})

    axs[0].hist(torch.log10(1 + x.sum(dim=1)), log=True, bins=np.linspace(0, 2., 81))
    axs[0].set_ylim([1, 600])
    axs[0].set_xlim([-0.1, 2.])
    axs[0].text(0.5, 300, "Number of active trials: %d/%d" % ((x.sum(dim=1) > 1).sum(), x.size(0)),
                horizontalalignment='left')
    if uncertainty is not None:
        axs[0].text(0.5, 100, "Uncertainty: %.4f" % (uncertainty.item()), horizontalalignment='left')

    prob = torch.softmax(logits, 0)
    colors = ['blue'] * len(class_names)
    colors[y] = 'red'
    axs[1].barh(class_names, prob.numpy(), align='center', color=colors)

    return fig


def generate_fingerprint(x, vmax=8):
    norm = plt.Normalize(vmin=0, vmax=vmax)
    colors = plt.cm.plasma(norm(x))
    colors = np.tile(colors, (5, 1, 1, 1)).transpose(1,3,0,2)[:, :3]
    return colors
