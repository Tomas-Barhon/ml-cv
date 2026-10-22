import argparse

import lab02_help
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument(
    "--data", default="hriby2.txt", type=str, help="Path to the file with data."
)


def fitness_cons(labels: np.ndarray, X: np.ndarray) -> float:
    """
    Consistency measure

    Write the body of this function. It should calculate the fitness (consistency)
    of a subset of features.

    Arguments:
    - 'labels' - Vector of true classes.
    - 'X' - Observations of a subset of features.

    Returns:
    - Fitness of the provided feature set.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    X = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])

    # Find unique rows
    # - 'C' - unique rows
    # - 'ix' - index to the first occurence in X for each unique row
    # - 'ic' - index to C for each row in X
    # - 'np.max(ic) = len(ix) - 1' ... indexing starts at 0
    # - 'C = X[ix, :]', 'X = C[ic, :]'
    # - 'M' - number of occurences for each unique row
    C, ix, ic, M = np.unique(
        X, axis=0, return_index=True, return_inverse=True, return_counts=True
    )

    # TODO: For each unique row find the number of inconsistent classifications.
    IC = 0
    unique_count = np.size(ix)
    for i in range(unique_count):
        # Classes of each unique row.
        row_classes = labels[ic == i]
        # Find the unique classes for the processed row.
        unique_row_classes, jx, jc, class_counts = np.unique(
            row_classes,
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        # TODO: Find the maximum of 'class_counts'.
        # TODO: Compute inconsistency of object 'i'.
        max_class_count = np.max(class_counts)
        IC += M[i] - max_class_count

    # TODO: Return the final value J according to the formula from the lecture.
    return 1 - IC / np.size(X, 0)


def fitness_cbfs(labels: np.ndarray, X: np.ndarray) -> float:
    """
    Correlation-based Feature selector

    Write the body of this function. It should calculate the fitness
    (Correlation-based Feature Selector) of a subset of features.

    Arguments:
    - 'labels' - Vector of true classes.
    - 'X' - Observations of a subset of features.

    Returns:
    - Fitness of the provided feature set.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    X = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])
    # Get the correlation coefficients.
    R = lab02_help.corrcoef(labels, X)
    # TODO: Compute the number of columns (features) in X.
    K = X.shape[1]

    # The upper triangular matrix of ones.
    ind_triangle = np.tri(K + 1, k=-1, dtype=bool).T

    # NOTE: Matlab uses a lower triangular matrix because it reads values column by column
    # whereas numpy does it row by row. The matrix R is symmetric.
    coefs = R[ind_triangle]
    rcf = np.mean(coefs[0:K])

    # TODO: Return the final value J according to the formula from the lecture.
    # - Think about how the formula changes when 'K == 1'.
    if K > 1:
        rff = np.mean(coefs[K:])
        return (K * rcf) / np.sqrt(K + (K * (K - 1) * rff))
    else:
        return 1


def fitness_icd(labels: np.ndarray, X: np.ndarray) -> float:
    """
    Interclass distance

    Write the body of this function. It should calculate the fitness
    (Interclass distance) of a subset of features.

    Arguments:
    - 'labels' - Vector of true classes.
    - 'X' - Observations of a subset of features.

    Returns:
    - Fitness of the feature set as the mean class distance.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    X = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])
    # Objects belonging to class 1.
    X_1 = X[labels == 1, :]
    # Objects belonging to class 0.
    X_0 = X[labels == 0, :]
    # Matrix of pair-wise distances of points in X_1 and X_0.
    d_X = cdist(X_1, X_0, "euclidean")
    D_X = np.mean(d_X)

    p_0 = len(X_0) / len(X)
    p_1 = len(X_1) / len(X)

    # TODO: Return the value of interclass distance according to the formula from the lecture.
    # - Compute the class probabilities as the fractions of objects with the particular classes.
    return p_0 * p_1 * D_X


def fitness_mi(labels: np.ndarray, X: np.ndarray) -> float:
    """
    Mutual information

    Write the body of this function. It should calculate the fitness
    (Mutual information) of a subset of features.

    Arguments:
    - 'labels' - Vector of the true class.
    - 'X' - Observations of a subset of features.

    Returns:
    - Mutual information of the feature subset and the classification.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    # - Renamed 'X' to 'Y' to match the formula from the slides.
    Y = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])
    # Follow the formula I(Y;X) = H(Y) - H(Y|X)
    # The following line computes the entropy of the whole set (Y in our formula).
    # - NOTE: 'X' in code is not the same as 'X' in the formula.
    entropy = get_entropy(Y)
    conditional_entropy = 0
    for label in np.unique(labels):
        p_label = len(labels[labels == label]) / len(labels)
        conditional_entropy += p_label * get_entropy(Y[labels == label])
    #
    # TODO: Compute the remaining terms of the formula.
    # - Class probabilities are the fractions of objects with the particular classes.
    # TODO: Return the value of mutual information according to the formula from
    # the lecture 'I(Y;X) = H(Y) - H(Y|X)'.
    return entropy - conditional_entropy


def get_entropy(X: np.ndarray) -> float:
    """
    Computes the entropy of a single data set.
    """
    # The following code computes the entropy of a single variable.
    # Find unique objects of the set.
    C, ix, ic, counts = np.unique(
        X, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    N = X.shape[0]
    # Compute the probability.
    probabilities = counts / N
    # Compute the entropy.
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def measure(labels: np.ndarray, features: np.ndarray) -> None:
    # Create a list with indices of features you want to use.
    selection = [1, 4, 5, 6]
    print("Selected feature indices: {}".format(selection))

    # X_tilde - set of data for these features.
    X = features[:, selection]

    # TODO: ===== Task 1 ('measure') =====
    # Modify functions computing different fitness measures within this script such that
    # they return the correct value.
    # You can start with one or two functions, continue with the other tasks and then
    # return to the other functions.
    J_cons = fitness_cons(labels, X)
    print(f"Consistency fitness:         {None if J_cons is None else f'{J_cons:.4f}'}")

    J_cbfs = fitness_cbfs(labels, X)
    print(f"Correlation based fitness:   {None if J_cbfs is None else f'{J_cbfs:.4f}'}")

    J_icd = fitness_icd(labels, X)
    print(f"Interclass distance fitness: {None if J_icd is None else f'{J_icd:.4f}'}")

    J_mi = fitness_mi(labels, X)
    print(f"Mutual information fitness:  {None if J_mi is None else f'{J_mi:.4f}'}")


def main(args: argparse.Namespace) -> None:
    # Load mushroom classification data.
    data = lab02_help.parse_text_file(args.data)

    # The first column in data matrix is the class.
    # The remaining columns 2:23 are features.
    # Each row is one observation.

    # Split data into two variables: 'labels' (class) and 'features' (features).
    labels = data[:, 0]
    features = data[:, 1:]
    measure(labels, features)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
