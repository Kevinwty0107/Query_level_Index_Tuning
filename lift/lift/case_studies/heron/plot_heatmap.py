import itertools
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

class HeatmapPlotter(object):

    @staticmethod      
    def plot(true_labels, pred_labels, no_labels,
                cmap = sns.cubehelix_palette(8, as_cmap=True)):
            conf_matrix = confusion_matrix(true_labels, pred_labels, 
                    labels=list(range(no_labels)))
            # normalise the matrix
            conf_matrix = conf_matrix.astype('float') /  \
                    conf_matrix.sum(axis=1)[:, np.newaxis]
            np.nan_to_num(conf_matrix, copy=False)
            plt.imshow(conf_matrix, interpolation='nearest', cmap = cmap)
            fmt = '.2f'
            thresh = conf_matrix.max() * 0.5
            for i, j in itertools.product(range(conf_matrix.shape[0]), 
                    range(conf_matrix.shape[1])):
                plt.text(j, i, format(conf_matrix[i, j], fmt),
                    horizontalalignment = "center",
                    color = "white" if conf_matrix[i,j] > thresh else "black")
                plt.tight_layout()
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
