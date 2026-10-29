import argparse
import os
import re
from typing import List

import lab03b_pca_svd
import matplotlib.pyplot as plt
import numpy as np
import skimage.io

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument(
    "--path",
    default="./centered/",
    type=str,
    help="Path to the 'face' dataset with root directory 'centered'.",
)
parser.add_argument(
    "--compare_idx",
    default=22,
    type=int,
    help="Index of the image selected for comparison.",
)
parser.add_argument(
    "--var_threshold",
    default=0.95,
    type=float,
    help="Threshold for cumulative explained variance.",
)
parser.add_argument(
    "--reconstruct_idx",
    default=11,
    type=int,
    help="Index of the image selected for gradual reconstruction.",
)


def main(args: argparse.Namespace) -> None:
    # TODO: Make sure that 'args.path' contains a correct path towards folder 'centered'
    # containing face dataset downloaded from moodle.
    training_folder = args.path
    regex = re.compile(r".*\..*\.pgm$")
    train_files = []
    dirs = os.listdir(args.path)
    for f in dirs:
        if regex.match(f):
            train_files.append(f)
    train_files = [training_folder + f for f in train_files]

    im = skimage.io.imread(train_files[0])
    H = im.shape[0]  # Image height
    W = im.shape[1]  # Image width
    M = len(train_files)  # Number of images

    # Read all images and transform them to vectors.
    flattened_imgs = np.zeros([M, H * W])
    for i in range(M):
        flattened_imgs[i] = skimage.io.imread(train_files[i]).flatten()

    # TODO: Compute the eigenvectors using SVD.
    # - use 'PCASVD' which you implemented in 'lab03b_pca_svd.py'.
    #
    # Name the computed variables as follows (so that the rest of the code works):
    # mu - the mean of the data
    # B - the eigenvectors in row form (1 row = 1 eigenvector)
    # X - the centred data (centred 'flattened_imgs')
    # S - explained variances
    pca = lab03b_pca_svd.PCASVD()
    pca.fit(flattened_imgs)
    mu = pca._mu
    B = pca._components
    S = pca._explained_variance
    X = flattened_imgs - mu

    # TODO: Look at several eigenfaces as images.
    shown_ind = [0, 1, 2, 3]
    pcs_as_images(shown_ind, B, H, W)

    # TODO: Determine the number of PCs to keep, look at the cumulative variances.
    # - Either plot the graph and read the value from there or use an automatic method.
    var_threshold = args.var_threshold
    k = np.argmax(np.cumsum(S) >= var_threshold) + 1

    # Keep k principal components.
    Bk = B[:k, :]

    # Compute the new coordinates in k dimensions.
    Xk = X @ Bk.T

    # Reconstruct the reduced data back to D dimensions.
    rec = mu + Xk @ Bk

    # Compare the original image and the reconstruction from principal components.
    img_idx = args.compare_idx
    compare_reconstruction(img_idx, flattened_imgs, rec, k, H, W)

    # Now, let's visualise the reconstructions from different number of principal components.
    # You can change the value of 'img_idx' to view the reconstruction of other images.
    # Also, you can modify the 'counts' list to see the reconstructions which interest
    # you.
    img_idx = args.reconstruct_idx
    counts = [1, 2, 5, 10, 20, 40, 80, 120]
    progressive_reconstruction(img_idx, counts, flattened_imgs, X, B, mu, H, W)


def pcs_as_images(shown_ind: List[int], B: np.ndarray, H: int, W: int) -> None:
    fig, ax = plt.subplots(1, len(shown_ind), figsize=(17, 5))
    for ax_idx in range(len(shown_ind)):
        ax_current: plt.Axes = ax[ax_idx]
        ax_current.set_title(f"PC {shown_ind[ax_idx]} reshaped back into an image")
        ax_current.imshow(np.reshape(B[shown_ind[ax_idx], :], [H, W]), cmap="gray")
    fig.tight_layout()
    plt.show()


def compare_reconstruction(
    img_idx: int, flattened_imgs: np.ndarray, rec: np.ndarray, k: int, H: int, W: int
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), subplot_kw={"aspect": "equal"})
    ax[0].set_title("Original image")
    ax[0].imshow(np.reshape(flattened_imgs[img_idx, :], [H, W]), cmap="gray")
    ax[1].set_title(f"Reconstruction from {k} PCs")
    ax[1].imshow(np.reshape(rec[img_idx, :], [H, W]), cmap="gray")
    fig.tight_layout()
    plt.show()


def progressive_reconstruction(
    img_idx: int,
    counts: List[int],
    flattened_imgs: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
    mu: np.ndarray,
    H: int,
    W: int,
) -> None:
    fig, ax = plt.subplots(
        1, len(counts) + 1, figsize=(20, 3), subplot_kw={"aspect": "equal"}
    )
    for i in range(len(counts)):
        count = counts[i]
        projected = X[img_idx, :] @ B[:count, :].T
        reconstructed = mu + projected @ B[:count, :]
        ax[i].imshow(np.reshape(reconstructed, [H, W]), cmap="gray")
        ax[i].set_axis_off()
        ax[i].set_title("{} PCs".format(count))
    ax[len(counts)].imshow(np.reshape(flattened_imgs[img_idx, :], [H, W]), cmap="gray")
    ax[len(counts)].set_axis_off()
    ax[len(counts)].set_title("Original")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
