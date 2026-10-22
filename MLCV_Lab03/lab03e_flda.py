
from typing import Self
import argparse
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sklearn.decomposition
import lab03_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument(
    "--mnist_data", default="mnist_train.npz", type=str,
    help="Path to the MNIST dataset (images or combined)."
)
parser.add_argument(
    "--mnist_labels", default=None, type=str,
    help="Path to the MNIST labels (if they are in a separate file)."
)
parser.add_argument(
    "--limit_histograms", default=4, type=int,
    help="Number of classes visualised in histogram analysis tool."
)

class FLDA:

    def __init__(self : Self) -> None:
        self._unique_labels : np.ndarray = None
        self._num_classes : int = None
        self._components : np.ndarray = None
        self._eigenvalues : np.ndarray = None
        self._mu : np.ndarray = None
        self._sv : np.ndarray = None
        self._sm : np.ndarray = None
        self._is_fitted = False

    def fit(self : Self, data : np.ndarray, labels : np.ndarray) -> None:
        """
        Computes FLDA on the given data and labels, storing the computed components,
        eigenvalues explaining the disciriminative power between components, and other
        algorithmic variables as class attributes. This function needs to be called
        before 'transform'.

        Arguments:
        - 'data' - A ndarray containing the data matrix of shape NxD.
        - 'labels' - A ndarray containg data labels of shape N.
        """
        # Get the unique labels and their count (It's probably 10 for MNIST but it
        # could be less).
        unique_labels, ix, ic = np.unique(labels, return_index=True, return_inverse=True)
        self._unique_labels = unique_labels
        self._num_classes = np.size(unique_labels)

        #X = data#imgs
        dims = data.shape[1]

        sv = np.zeros([dims, dims])
        sm = np.zeros([dims, dims])

        # Compute the mean of the data (assumes that images are in rows).
        self.mu = np.mean(data, axis=0)

        for j in unique_labels:
            # Get the samples for the current class (assumes that images are in rows).
            data_j = data[(labels==j), :]
            # TODO: Compute the mean of the data belonging to the class.
            mu_j = None
            # TODO: Compute the size of the class (number of images in the class).
            nj = None
            # TODO: Update the value of the matrix Sv.
            sv = None
            # TODO: Update the value of the matrix Sm.
            # - Look carefully at the shapes of the mean variables.
            sm = None

        self._sv = sv
        self._sm = sm
        # Solve the general eigenvalue problem:
        # - We use pseudo-inverse to avoid problems with singular matrices. This might produce
        #   complex eigenvalue decomposition and so we take only the real part of the result.
        #   We lose some information this way, but it is a straight-forward solution.
        M = np.linalg.pinv(sv) @ sm
        eigvals, eigvecs = scipy.linalg.eig(M)
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        # Sort the eigenvectors.
        desc_order = np.argsort(-eigvals)
        eigvals = eigvals[desc_order]
        # The matrix produced by scipy.linalg.eig is in column-major form.
        eigvecs = eigvecs[:, desc_order]
        # Keep at most (c - 1) eigenvectors.
        eigvecs = eigvecs[:, :self._num_classes-1]
        self._components = eigvecs
        self._eigenvalues = eigvals[:self._num_classes-1]

        self._is_fitted = True

    def transform(self : Self, data : np.ndarray) -> np.ndarray:
        """
        Computes the projection of the given data matrix into the coordinate system
        defined by the FLDA components computed in 'fit'.
        """
        if not self._is_fitted:
            raise ValueError("Calling PCASVD 'transform' before 'fit'.")
        # Project the data.
        projected = data @ self._components
        return projected
    
    @property
    def unique_labels(self : Self) -> np.ndarray:
        """The unique labels appearing in the data as discovered in the 'fit' fucntion."""
        return self._unique_labels

def main(args : argparse.Namespace):
    # TODO: Load MNIST data, which we prepared in the first practical.
    dataset = lab03_help.MnistDataset(args.mnist_data, labels=args.mnist_labels)
    imgs = dataset.imgs # Change if your image data block is called differently.
    labels = dataset.labels # Change if your label data block is called differently.

    # Select and use only the first 1000 images and labels. Make sure that the images are
    # in rows. Store the selected part of the dataset back in 'imgs' and 'labels'.
    imgs = imgs[:1000, :]
    labels = labels[:1000]
    
    flda = FLDA()
    flda.fit(imgs, labels)
    proj_flda = flda.transform(imgs)

    # Compute PCA using scikit-learn.
    pca = sklearn.decomposition.PCA(n_components=None)
    pca.fit(imgs)

    # Compute the new coordinates.
    # Note that sklearn.decomposition.PCA.transform centres the data.
    proj_pca = imgs @ pca.components_.T
    #proj_pca = pca.transform(imgs)

    # Plot the data transformed by PCA and FLDA together with the explained variance of PCA.
    pca_expl_var_ratio = pca.explained_variance_ratio_
    compare_transformations(labels, flda.unique_labels, proj_flda, proj_pca, pca_expl_var_ratio)

    # TODO: You can change the limit of the number of classes which will be shown
    # in the histograms. There is 10 in total, so try, for instance, 2, 3, 5, 10.
    # Note, that the histograms are very cluttered for more than 4 classes.
    class_count_limit = args.limit_histograms
    selected_examples = proj_pca[labels < class_count_limit, :3]
    #selected_examples = proj_flda[labels < class_count_limit, :3]

    lab03_help.plot_histograms(selected_examples, labels[labels < class_count_limit])

def compare_transformations(
        labels : np.ndarray, cnames : np.ndarray, proj_flda : np.ndarray, proj_pca : np.ndarray,
        explained_variance : np.ndarray
    ) -> None:
    # Plot the data transformed by PCA and FLDA together with the explained variance of PCA.
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'brown', 'coral', 'gold']
    fig = plt.figure("PC Visualisation", figsize=(14, 5))
    ax1 : plt.Axes = fig.add_subplot(131, projection="3d")
    ax2 : plt.Axes = fig.add_subplot(132, projection="3d")
    ax3 : plt.Axes = fig.add_subplot(133, aspect="auto")
    ax1.set_title("The first 3 PCs of FLDA")
    for i in cnames:
        digit_indices = labels == i
        ax1.scatter(
            proj_flda[digit_indices, 0], proj_flda[digit_indices, 1], proj_flda[digit_indices, 2],
            marker='o', color=colors[i], label="{}".format(i)
        )
    ax1.legend()
    ax2.set_title("The first 3 PCs of PCA")
    for i in cnames:
        digit_indices = labels == i
        ax2.scatter(
            proj_pca[digit_indices, 0], proj_pca[digit_indices, 1], proj_pca[digit_indices, 2],
            marker='o', color=colors[i], label="{}".format(i)
        )
    ax2.legend()
    ax3.set_title("PCA Explained variance")
    ax3.axhline(color='black', ls='--')
    ax3.axvline(color='black', ls='--')
    ax3.plot(explained_variance, color='red', label="Explained")
    ax3.plot(np.cumsum(explained_variance), color='blue', label="Cumulative")
    ax3.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
