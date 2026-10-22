
from typing import Tuple, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt

class MnistDataset:
    """
    Loads the MNIST data saved in .npy or .npz files.

    If the 'labels' argument is left as None then the class assumes that the file
    in 'data' is .npz and creates attributes, with the same name as specified
    during the file creation, containing the respective numpy arrays.

    If the 'labels' argument is set to a string path then the class assumes that
    the files were saved as .npy and it will create two attributes: 'imgs' which
    contains the contents of the 'data' file and 'labels' with the contents of
    the 'labels' file.

    If you chose to save the arrays differently then you might have to modify
    this class or write your own loader.
    """

    def __init__(self, data : str = "mnist_train.npz", labels : str = None):

        if not os.path.exists(data):
            raise ValueError("Requested mnist data file not found!")
        if (labels is not None) and (not os.path.exists(labels)):
            raise ValueError("Requested mnist label file not found!")

        if labels is None:
            dataset = np.load(data)
            for key, value in dataset.items():
                setattr(self, key, value)
        else:
            self.imgs = np.load(data)
            self.labels = np.load(labels)

def plot_data_with_pcs(
        data : np.ndarray,
        labels : np.ndarray,
        mu : Tuple[float, float] = None,
        pcs : np.ndarray = None
    ) -> None:
    """
    Plots 2D data together with the axes formed by the principal components.
    Data (and mu) has to be in row form, the 'pcs' matrix has to have
    the principal components in columns.

    Arguments:
    - 'data' - Data points given in rows.
    - 'labels' - Numerical labels of the given data points.
    - 'mu' - Mean of the data points.
    - 'pcs' - Principle components given in columns.
    """
    axes = np.asarray([6, 3]) * pcs if pcs is not None else None
    fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
    ax.set_title(f"Data coloured according to labels{"with PCs" if pcs is not None else ""}")
    ax.scatter(data[:, 0], data[:, 1], c=labels)
    if pcs is not None:
        ax.plot(
            [mu[0], mu[0] + axes[0, 0]], [mu[1], mu[1] + axes[1, 0]], color='blue', linewidth=5
        )
        ax.plot(
            [mu[0], mu[0] + axes[0, 1]], [mu[1], mu[1] + axes[1, 1]], color='green', linewidth=5
        )
    fig.tight_layout()
    plt.show()

def plot_histograms(
        points : np.ndarray,
        labels : np.ndarray,
        colors : Sequence[str] = None
    ) -> None:
    """
    Plots the dataset together with histograms showing the distribution of the individual
    classes.
    If there is more than 10 classes, attribute 'colors' has to be specified and it needs
    to define colour for every observation.

    Arguments:
    - 'points' - Data points given in rows.
    - 'labels' - Labels of the given data points.
    - 'colors' - Optional list of string values marking the colours of the classes.
    """
    labels = labels.flatten()
    classes, idx = np.unique(labels, return_index=True)
    classes = np.asarray(classes, np.int32)

    delta = np.mean(np.max(points, axis=0) - np.min(points, axis=0)) / 15
    bin_count = (np.max(points, axis=0) - np.min(points, axis=0)) / delta
    bin_count = np.asarray((bin_count / np.max(bin_count)) * 8 + 12, np.int32)
    bin_values = [
        [
            np.histogram(
                points[labels == k, i],
                bin_count[i],
                range=(np.min(points[:, i]), np.max(points[:, i]))
            ) for k in classes
        ] for i in range(points.shape[1])
    ]
    
    colour_selection = [
        'blue', 'red', 'green', 'cyan', 'magenta',
        'yellow', 'black', 'brown', 'coral', 'gold'
    ]
    if colors is None:
        colors = np.asarray(colour_selection)[np.asarray(labels, np.int8) - 1]
    else:
        colour_selection = colors[idx]
    
    fig, ax = plt.subplots(
        points.shape[1], points.shape[1], figsize=(11, 8), subplot_kw={'aspect': 'auto'}
    )
    fig.suptitle("Discriminability of principle components")
    for i in range(points.shape[1]):
        for j in range(points.shape[1]):
            ax_ij : plt.Axes = ax[i, j]
            if i == j:
                for k in range(len(bin_values[i])):
                    ax_ij.step(
                        bin_values[i][k][1][:-1],
                        bin_values[i][k][0],
                        where='post',
                        c=colour_selection[classes[k] - 1]
                    )
            else:
                ax_ij.scatter(points[:, j], points[:, i], c=colors)
    fig.tight_layout()
    plt.show()

def plot_variance(
        ax : plt.Axes, explained : np.ndarray, cumulative : np.ndarray, name : str
    ) -> None:
    """
    Plots a scree graph consisting of explaiend variances and their cumulative sum
    given as arguments into a pyplot Axes object.

    Arguments:
    - 'ax' - plt.Axes object used for plotting.
    - 'explained' - The explained variances.
    - 'cumulative' - The cumulative explained variances.
    + 'name' - The title of the Axes object.
    """
    ax.set_title(name)
    ax.axhline(color='black', ls='--')
    ax.axvline(1.0, color='black', ls='--')
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.plot(
        range(1, explained.size + 1), explained, color='red', marker='o', label="Explained"
    )
    ax.plot(
        range(1, cumulative.size + 1), cumulative, color='blue', marker='o', label="Cumulative"
    )
    ax.legend()
