import argparse
import time
from typing import Self

import lab03a_pca_eigen
import numpy as np
import scipy.io
import sklearn.decomposition

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument(
    "--data",
    default="hald",
    type=str,
    choices=["hald", "cancer"],
    help="Name of the dataset to use: 'hald' or 'cancer'.",
)


class PCASVD:
    def __init__(self):
        self._components: np.ndarray = None
        self._explained_variance: np.ndarray = None
        self._mu: np.ndarray = None
        self._is_fitted = False

    def fit(self: Self, data: np.ndarray) -> None:
        """
        Computes the singular value decomposition on data such that the result
        represents the components and explained variances of PCA.

        Arguments:
        - 'data' - NxD data matrix where N is the number of observation and D is
                   the number of features.
        """
        # TODO: Compute the mean for each feature.
        self._mu = np.mean(data, axis=0)

        # TODO: Construct the matrix Y from the lecture.
        Y = np.sqrt(data.shape[0] - 1) * (data - self._mu)

        # TODO: Compute SVD of Y and assign the correct values to 'self._components'.
        # and 'self._explained_variance' (do not forget to normalise the variance).
        # - The right eigenvectors are situated in rows.
        U, S, V = np.linalg.svd(Y, full_matrices=False)
        self._components = V
        self._explained_variance = (S**2) / np.sum(S**2)

        # Let's remember that we have already computed valid principal components.
        self._is_fitted = True

    def transform(self: Self, data: np.ndarray) -> np.ndarray:
        """
        Transforms the given 'data' into the coordinate system given by the principal
        components computed in the 'fit' function.
        """
        if not self._is_fitted:
            raise ValueError("Calling PCASVD 'transform' before 'fit'.")
        # TODO: Project the data onto PCs.
        projected = (data - self._mu) @ self._components.T
        return projected

    @property
    def components(self: Self) -> np.ndarray:
        """
        Returns the ordered principal components computed in the 'fit' function.
        The components are in the rows of the returned ndarray.
        """
        return self._components

    @property
    def explained_variance(self: Self) -> np.ndarray:
        """Returns the ordered explained variance of the principal components."""
        return self._explained_variance

    @property
    def mu(self: Self) -> np.ndarray:
        """Returns the mean of the training data computed in the 'fit' function."""
        return self._mu


def main(args: argparse.Namespace) -> None:
    if args.data == "hald":
        hald = scipy.io.loadmat("hald.mat")
        data = hald["ingredients"]
    elif args.data == "cancer":
        # TODO: Measure the execution time of each method on 'cancer' dataset after
        # you finish implementation of SVD PCA.
        # - This dataset is much larger than the first so it will be slower
        #   to compute.
        # - The function 'time.time()' returns the current time in seconds.
        # - NOTE: There are more dimensions in the data than there are examples,
        #   therefore we can get only that many principal components - this results
        #   in 'random' vectors and angles in the last dimensions.
        ovariancancer = scipy.io.loadmat("ovariancancer.mat")
        data = ovariancancer["obs"]

    # (EIG) TODO: PCA using eigenvalues of covariance matrix
    # - Take your solution to the exercise in lab03a
    pca_eig = lab03a_pca_eigen.PCAEigen()
    pca_eig.fit(data)
    B = pca_eig.components

    # (SVD) TODO: Implement the body of function 'fit' and 'transform' of the 'PCASVD'
    # class at the top of the script.
    # - The principal components produced by np.linalg.svd are in rows.
    pca_svd = PCASVD()
    pca_svd.fit(data)
    V = pca_svd.components

    # (PCA) TODO: Analyse the scikit-learn PCA method
    # - The computed principal components are in rows
    # - n_components=None means that we want all of the components
    pca = sklearn.decomposition.PCA(n_components=None)
    pca.fit(data)

    # TODO: The builtin principal components and explained variances are stored in
    # variables 'components_' and 'explained_variance_', similar to our implementation.
    C = pca.components_
    explained_variance = pca.explained_variance_
    # Projection is computed the same way.
    projected = pca.transform(data)

    # TODO: Find the mutual angles between vectors B, V and C.
    # How is the scalar product of vectors related to their angle?
    # - You should know this from linear algebra or any CG/geometry course.
    with np.printoptions(precision=3, suppress=True):
        # TODO: Uncomment the print lines and replace underscores '_' with
        # the appropriate matrices ('B', 'B.T', 'V', 'V.T', 'C', 'C.T').
        # - 'B' has PCs in columns, 'V' and 'C' have PCs in rows.
        print("Angles between PCAEigen (B) and PCASVD (V)")
        print(np.abs(np.arccos(np.clip(V @ B, -1, 1))) / np.pi * 180)
        print("Angles between PCASVD (V) and Sklearn (C)")
        print(np.abs(np.arccos(np.clip(V @ C.T, -1, 1))) / np.pi * 180)
        print("Angles between PCAEigen (B) and Sklearn (C)")
        print(np.abs(np.arccos(np.clip(C @ B, -1, 1))) / np.pi * 180)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
