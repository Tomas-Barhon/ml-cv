import load_mnist
import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    # 1. Use 'load_mnist.loadMnist' function to read 60000 training images and class labels
    #    of MNIST digits (files 'train-images.idx3-ubyte' and 'train-labels.idx1-ubyte').
    imgs_train, labels_train = load_mnist.load_mnist(
        "train-images.idx3-ubyte", "train-labels.idx1-ubyte", -1, 0, 0
    )

    # 2. Preview one image from each class.
    seen_classes = set()
    for i in range(len(labels_train)):
        label = labels_train[i]
        if label not in seen_classes:
            plt.imshow(imgs_train[i], cmap="gray")
            plt.title(f"Label: {label}")
            plt.axis("off")
            plt.show()
            seen_classes.add(label)
        if len(seen_classes) == 10:
            break

    # 3. Transform the image data, such that each image forms a row vector,
    #    - NOTE: Math in the lectures assumes the column-based format, exercises will assume
    #      the row-based format (Row-based format is used by most Python libraries). This means
    #      that we will have to "transpose" formulae before using them.
    imgs_train = imgs_train.reshape(imgs_train.shape[0], -1)
    # 4. Save the image matrix and labels in a numpy .npz file.
    #    - Use 'np.savez_compressed' with attribute names 'imgs' for images and 'labels' for labels.
    #    - Loading is done using 'np.load' which returns a dictionary with attribute names as keys.
    np.savez_compressed("mnist_train.npz", imgs=imgs_train, labels=labels_train)

    # 5. Do the same for 10000 test digits (files 't10k-images.idx3-ubyte' and
    #    't10k-labels.idx1-ubyte')
    #    - Both files (training and testing set) will be used during the semester.

    # 6. Now, try to load the created files using 'np.load', display some of the images and print
    #    their respective labels.


if __name__ == "__main__":
    main()
