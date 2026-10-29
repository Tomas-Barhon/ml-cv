import argparse

import lab03_help
import lab03a_pca_eigen
import lab03b_pca_svd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sklearn.decomposition

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument(
    "--task",
    default="scree",
    type=str,
    choices=["scree", "pcselection"],
    help="Task to execute: 'scree' or 'pcselection'.",
)


def scree_graph(args: argparse.Namespace) -> None:
    """This is the main task of this exercise."""
    # Load the data.
    hald = scipy.io.loadmat("hald.mat")
    data = hald["ingredients"]

    # TODO: Select the explained variances and compute the cumulative explained variances
    # for all three PCA computations methods shown in lab03b (numpy.cumsum might be useful
    # in this regard).
    # - The results are plotted as 'scree graphs' for an easier inspection.
    #
    # Identify the number of 'highly' contributing components from a scree graph.

    # Manual PCA
    pca_eig = lab03a_pca_eigen.PCAEigen()
    pca_eig.fit(data)
    eig_explained = pca_eig._explained_variance
    eig_cumulative = pca_eig._explained_variance.cumsum()

    # Scikit-learn PCA
    pca = sklearn.decomposition.PCA(n_components=None)
    pca.fit(data)
    skl_explained = pca.explained_variance_ratio_
    skl_cumulative = pca.explained_variance_ratio_.cumsum()

    # SVD PCA
    pca_svd = lab03b_pca_svd.PCASVD()
    pca_svd.fit(data)
    svd_explained = pca_svd._explained_variance
    svd_cumulative = pca_svd._explained_variance.cumsum()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={"aspect": "auto"})
    lab03_help.plot_variance(ax[0], eig_explained, eig_cumulative, "Eigen PCA")
    lab03_help.plot_variance(ax[1], skl_explained, skl_cumulative, "Scikit-learn PCA")
    lab03_help.plot_variance(ax[2], svd_explained, svd_cumulative, "SVD PCA")
    fig.tight_layout()
    plt.show()


def pc_selection(args: argparse.Namespace) -> None:
    """This is a secondary task of this exercise."""
    # TODO: Find the number of 'highly' contributing PCs - compute the number of PCs
    # required to explain 95% of the variance in the data.
    # - use PCA implemented in scikit-learn (sklearn.decomposition.PCA) and the following data.
    ovariancancer = scipy.io.loadmat("ovariancancer.mat")
    data = ovariancancer["obs"]
    var_threshold = 0.95

    feat_count = None
    print(
        f"Number of features for {int(var_threshold * 100)}% explained variance: {feat_count}"
    )


def main(args: argparse.Namespace) -> None:
    # Select task to execute.
    tasks = {
        "scree": scree_graph,
        "pcselection": pc_selection,
    }
    if args.task not in tasks.keys():
        raise ValueError("Unrecognised task: '{}'.".format(args.task))
    tasks[args.task](args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
