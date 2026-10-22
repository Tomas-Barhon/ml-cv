
import argparse
from typing import Self
import numpy as np
import lab03_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=42, type=int, help="Seed for RNG.")
parser.add_argument("--points", default=100, type=int, help="Number of points to generate.")

class PCAEigen:

    def __init__(self):
        self._components = None
        self._explained_variance = None
        self._mu = None
        self._is_fitted = False

    def fit(self : Self, data : np.ndarray) -> None:
        """
        This function is our manual implementation of the principal component analysis.
        It can be used to find the components from a 2D matrix of observations x features.
        The principal components and explained variance are stored as class properties

        Arguments:
        - 'data' - NxD data matrix representing N observations with D features.
        """
        # TODO: Compute the mean value for each feature and store it in
        # the variable 'mu' (should be a vector).
        self._mu = None

        # TODO: Centre the 'data' matrix (centering means subtracting the mean) - make
        # sure that the 'data' object is not modified by this subtraction.
        data_centered = None

        # TODO: Compute the covariance matrix and name it 'sig'.
        sig = None

        # Compute eigenvectors V and eigenvalues D (let's use numpy.linalg.eig).
        D, V = np.linalg.eig(sig)
        # 'eig' returns complex values with 0 imaginary part in some versions.
        D = np.real(D)
        V = np.real(V) 
        # Sort the eigenvalues in descending order.
        desc_order = np.argsort(-D)
        self._explained_variance = D[desc_order]
        # Normalise the explained variance.
        self._explained_variance = self._explained_variance / np.sum(self._explained_variance)
        # Create the basis vector matrix consisting of the ordered components, principal
        # components are in the columns of V.
        self._components = V[:, desc_order]

        # Let's remember that we have already computed valid principal components.
        self._is_fitted = True

    def transform(self : Self, data : np.ndarray) -> np.ndarray:
        """
        Transforms the given 'data' into the coordinate system given by the principal
        components computed in the 'fit' function.
        """
        if not self._is_fitted:
            raise ValueError("Calling PCAEigen 'transform' before 'fit'.")
        # TODO: Project the data onto PCs (principal components).
        projected = None
        return projected

    @property
    def components(self : Self) -> np.ndarray:
        """
        Returns the ordered principal components computed in the 'fit' function.
        The components are in the columns of the returned ndarray.
        """
        return self._components
    
    @property
    def explained_variance(self : Self) -> np.ndarray:
        """Returns the ordered explained variance of the principal components."""
        return self._explained_variance
    
    @property
    def mu(self : Self) -> np.ndarray:
        """Returns the mean of the training data computed in the 'fit' function."""
        return self._mu


def main(args : argparse.Namespace) -> None:
    # Random number generator for reproducibility.
    generator = np.random.RandomState(args.seed)

    # Prepare 2D data
    # Data of class 1.
    N1 = args.points
    mu1 = [1, 1]
    sigma1 = [[1, 1.5], [1.5, 5]]
    dat1 = generator.multivariate_normal(mu1, sigma1, N1)

    # Data of class 2.
    N2 = args.points
    mu2 = [4, 6]
    sigma2 = [[1, 0], [0, 1]]
    # TODO: (Optional) Try different data parameters after completing the exercise.
    #mu2 = [4, 1]
    #sigma2 = [[1, 1.5], [1.5, 3]]
    dat2 = generator.multivariate_normal(mu2, sigma2, N2)

    # Combined data.
    data = np.vstack([dat1, dat2])
    labels = np.hstack([np.ones(N1), 2 * np.ones(N2)])

    # Let us use our implementation of PCA.
    # TODO: Implement the PCAEigen class at the top of the script.
    pca_eig = PCAEigen()
    pca_eig.fit(data)
    projected = pca_eig.transform(data)

    # Plot the new coordinate axes (principal components).
    lab03_help.plot_data_with_pcs(data, labels, pca_eig.mu, pca_eig.components)

    # Look at histograms of projections onto PC1 and PC2.
    lab03_help.plot_histograms(projected, labels)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
