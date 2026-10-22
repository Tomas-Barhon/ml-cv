import argparse

import lab02_help
import numpy as np
from lab02_features import fitness_cbfs, fitness_cons, fitness_icd, fitness_mi

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
# This exercise has 2 tasks:
# - 'one-step' - Use the implemented fitness measures to compute one-step forward feature
#                selection.
# - 'sequential' - Use the implemented fitness measures to compute sequential forward
#                  feature selection.
parser.add_argument(
    "--task",
    default="one-step",
    type=str,
    choices=["one-step", "sequential"],
    help="Performed task: 'one-step', 'sequential'.",
)
parser.add_argument(
    "--data", default="hriby2.txt", type=str, help="Path to the file with data."
)
parser.add_argument(
    "--select", default=3, type=int, help="Number of selected features."
)


def task_one_step(labels: np.ndarray, features: np.ndarray, num_selected: int) -> None:
    # TODO: ===== Task 2 ('one-step') =====
    # Implement the One-step forward selection.
    #
    # Evaluate individual features, select top 3.
    # - 'np.argsort' can be used to find indices of a sorted array (to find the indices of
    #   features with the highest scores).
    # - 'np.sort' can be used to just sort an array of values, but 'np.argsort' is
    #   generally more useful.
    # Compare the sets of features chosen with different fitness functions.
    num_features = features.shape[1]
    fitness_arr = np.zeros([4, num_features])
    for feat_id in range(num_features):
        fitness_arr[:, feat_id] = (
            fitness_cbfs(labels, features[:, feat_id]),
            fitness_cons(labels, features[:, feat_id]),
            fitness_icd(labels, features[:, feat_id]),
            fitness_mi(labels, features[:, feat_id]),
        )

    fitness_ind = np.argsort(-fitness_arr)[:num_selected]
    selection = fitness_ind[:, :num_selected]
    print(selection)


def task_sequential(
    labels: np.ndarray, features: np.ndarray, num_selected: int
) -> None:
    # TODO: ===== Task 3 ('sequential') =====
    # Implement the Sequential forward selection.
    #
    # Evaluate individual features, select top 3.
    # - 'np.argmax' can be used to find the feature with the highest score.
    # - 'np.nanargmax' is 'np.argmax', which ignores NaN (Not a Number) values.
    # Compare the sets of features chosen with different fitness functions.
    fitness_measure = {0: fitness_cons, 1: fitness_cbfs, 2: fitness_icd, 3: fitness_mi}
    selected = np.zeros([4, num_selected], dtype=int)

    print(selected)


def main(args: argparse.Namespace) -> None:
    # Load mushroom classification data.
    data = lab02_help.parse_text_file(args.data)

    # The first column in data matrix is the class.
    # The remaining columns 2:23 are features.
    # Each row is one observation.

    # Split data into two variables: 'labels' (class) and 'features' (features).
    labels = data[:, 0]
    features = data[:, 1:]

    tasks = {"one-step": task_one_step, "sequential": task_sequential}
    if args.task not in tasks:
        raise ValueError("Task '{}' is not recognised!".format(args.task))
    tasks[args.task](labels, features, args.select)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
