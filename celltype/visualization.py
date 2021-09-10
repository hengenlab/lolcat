import itertools
import numpy as np

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
