import argparse

import lab04_help
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument(
    "--threshold", default=0.4, type=float, help="Threshold for binarisation."
)


class BernoulliNB:
    def __init__(self, threshold=None) -> None:
        """
        Creates a bernoulli naive bayes classifier with similar signature and methods as
        scikit-learn classes. In short, 'fit' trains the classifier on a training set with
        its labels, 'likelihood' returns probabilities of belonging to the training
        classes, and 'predict' returns the final predictions.

        Arguments:
        - 'threshold' - Specifies a threshold used before training and prediction. If not
          specified then data passed to 'fit', 'likelihood', and 'predict' have to be
          binary.
        """
        self._threshold = threshold
        self._training_completed = False

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Trains the Bernoulli Naive Bayes classifier."""
        # Binarise data if the threshold was selected.
        data = (
            data
            if self._threshold is None
            else np.asarray(data > self._threshold, float)
        )

        # Store the number of training samples and the number of features for later.
        num_features = data.shape[1]
        num_samples = data.shape[0]

        # Preparation for Bayes - count the number of classes.
        self.class_names = np.unique(labels)
        self.num_classes = np.size(self.class_names)

        # NOTE: Now, we have black-and-white images of digits and the pixel values are
        # either 0 or 1. Therefore, we are going to examine the probabilities of
        # the individual features (pixels in the image) having the value 1. Then we can
        # compute probabilities for the value 0 simply as the complement to one.

        # Probability that the feature k (the pixel k) equals 1 (is white) in
        # the class c1 (P(x_k=1|omega_c1))
        self.p_1_class = np.zeros([num_features, self.num_classes])
        # Probability of class c1 (P(omega_c1)) (Prior)
        self.p_class = np.zeros([self.num_classes])

        for class_idx in range(np.size(self.class_names)):
            # Select all samples from the current class self.cnames[clIdx]
            Xi = data[labels == self.class_names[class_idx], :]
            # TODO: Compute the number of objects from the class where feature k has
            # the value 1.
            # - It should be a vector with one value for each feature.
            N_ki = np.sum(Xi, axis=0)
            # print(N_ki)
            # TODO: Get the number of objects from the class.
            Ni = Xi.shape[0]
            # TODO: Compute the probability of pixels having the value 1 and use Laplace
            # smoothing with value alpha=1.
            self.p_1_class[:, class_idx] = (N_ki + 1) / (Ni + 2)
            # TODO: Compute the probability of the class (Prior).
            self.p_class[class_idx] = Ni / num_samples
        # We need the probability that feature k equals 0 in the class as well.
        self.p_0_class = 1 - self.p_1_class

        # NOTE: For numerical reasons we will be working with logarithms.
        # If the probabilities can be equal to zero then it is necessary to add a small
        # value, i.e. np.spacing(1), but this will not happen in our case - why?

        # TODO: Compute the natural logarithm of probabilities that feature k equals 1.
        self.log_p_1_class = np.log(self.p_1_class)
        # TODO: Do the same for p_0_class.
        self.log_p_0_class = np.log(self.p_0_class)
        # TODO: Aaaand do the same for the prior probabilities.
        self.log_p_class = np.log(self.p_class)

        self._training_completed = True

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Computes the probability of belonging to each class for all data samples."""
        if not self._training_completed:
            raise RuntimeError(
                (
                    "BernoulliNB method 'likelihood' was called before 'fit'. "
                    "Please ensure that the model finished training before prediction."
                )
            )
        # Binarise data if the threshold was selected.
        data = (
            data
            if self._threshold is None
            else np.asarray(data > self._threshold, float)
        )

        # The "probabilities" of belonging to classes for every object.
        class_likelihood = np.zeros([data.shape[0], self.num_classes])
        for j in range(data.shape[0]):
            for c in range(self.num_classes):
                # TODO: Compute the likelihood (decision function) for every image and class.
                class_likelihood[j, c] = (
                    np.sum(
                        data[j, :] * self.log_p_1_class[:, c]
                        + (1 - data[j, :]) * self.log_p_0_class[:, c]
                    )
                    + self.log_p_class[c]
                )

        return class_likelihood

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Computes the class prediction for each data sample."""
        # This method should result in the final class prediction which is the one with
        # highest likelihood - the right class is probably the most probable one.
        return np.argmax(self.likelihood(data), axis=1)


def main(args: argparse.Namespace):
    # Load the MNIST data prepared in the first practical. Use the MnistDataset class.
    # You might have to rename the variables in the following code depending on how you
    # saved the data.
    # TODO: Ensure that the mnist arrays have the following shape:
    # - train.imgs as       60000x784
    # - train.labels as     1x60000 (1D vector of 60000 elements)
    # - test.imgs as        10000x784
    # - test.labels as      1x10000 (1D vector of 10000 elements)
    train = lab04_help.MnistDataset("mnist_train.npz")
    test = lab04_help.MnistDataset("mnist_test.npz")

    # Let us binarise the images (each pixel will be either 0 or 1)
    # - Bernoulli classifier works only with binary data.
    # - Default threshold assumes data values in range from 0 to i.
    # - Both training and testing data must be binarised.
    # TODO: Complete the 'BernoulliNB.fit()' method.
    bin_train_imgs = np.asarray(train.imgs > args.threshold, float)
    bin_test_imgs = np.asarray(test.imgs > args.threshold, float)
    bayes_classifier = BernoulliNB()
    bayes_classifier.fit(bin_train_imgs, train.labels)

    # TODO: Now, complete the body of the function 'BernoulliNB.likelihood()' used by
    # 'BernoulliNB.predict()' to get the final class predictions.
    train_predictions = bayes_classifier.predict(bin_train_imgs)
    test_predictions = bayes_classifier.predict(bin_test_imgs)

    # TODO: Compute the accuracy of the classification.
    # - how many of the selected class indices match the original labels?
    train_accuracy = np.sum(train_predictions == train.labels) / len(train.labels)
    test_accuracy = np.sum(test_predictions == test.labels) / len(test.labels)

    # TODO: Try to change the binarisation threshold.
    # - Does the number of correctly classified images change?
    print(f"Train set accuracy: {100 * train_accuracy:.2f}%")
    print(f"Test set accuracy: {100 * test_accuracy:.2f}%")

    # Let's look at images created from the probabilities, they should resemble the digits.
    visualize_probabilities(
        bayes_classifier, int(np.round(np.sqrt(bin_train_imgs.shape[1])))
    )


def visualize_probabilities(bayes_classifier: BernoulliNB, side: int) -> None:
    fig, ax = plt.subplots(
        1,
        len(bayes_classifier.class_names),
        figsize=(20, 3),
        subplot_kw={"aspect": "equal"},
    )
    for class_idx in range(len(bayes_classifier.class_names)):
        ax_idx: plt.Axes = ax[class_idx]
        ax_idx.imshow(
            np.reshape(
                bayes_classifier.p_1_class[:, bayes_classifier.class_names[class_idx]],
                [side, side],
            ),
            cmap="gray",
        )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
