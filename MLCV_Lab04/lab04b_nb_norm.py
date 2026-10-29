
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import lab04_help

class GaussianNB:

    def __init__(self) -> None:
        """
        Creates a Gaussian naive bayes classifier with similar signature and methods as
        scikit-learn classes. In short, 'fit' trains the classifier on a training set
        with its labels, 'likelihood' computes the probabilities of belonging to each
        class, and 'predict' returns the final class predictions.
        """
        self._training_completed = False

    def fit(self, data : np.ndarray, labels : np.ndarray) -> None:
        """Trains the Bernoulli Naive Bayes classifier."""
        # Store the number of training samples and the number of features for later.
        num_features = data.shape[1]
        num_samples = data.shape[0]

        # Preparation for Bayes - count the number of classes.
        self.class_names = np.unique(labels)
        self.num_classes = np.size(self.class_names)

        # Probability of classes.
        self.p_class = np.zeros([self.num_classes])
        # Mean of each feature for the classes.
        self.mean_class = np.zeros([num_features, self.num_classes])
        # Standard deviation of each feature for the classes.
        self.std_class = np.zeros([num_features, self.num_classes])

        for class_idx in range(np.size(self.class_names)):
            # Select all samples from the current class self.cnames[class_idx].
            Xi = data[labels == self.class_names[class_idx]]
            # TODO: Compute the mean and standard deviation for every feature(pixel).
            # # - np.mean, np.std
            self.mean_class[:, class_idx] = None
            self.std_class[:, class_idx] = None
            # TODO: Compute the class probability (Prior).
            self.p_class[class_idx] = None
        
        self._training_completed = True

    def likelihood(self, data : np.ndarray) -> np.ndarray:
        if not self._training_completed:
            raise RuntimeError((
                "GaussianNB method 'likelihood' was called before 'fit'. "
                "Please ensure that the model finished training before prediction."
            ))
        
        # The 'probabilities' of belonging to classes for every object.
        class_likelihood = np.zeros([data.shape[0], self.num_classes])

        eps = np.spacing(1)
        for class_idx in range(self.num_classes):
            mu = self.mean_class[:, class_idx]
            std = self.std_class[:, class_idx]
            # Evaluate the pdf of the data.
            # - If std==0 then pdf computation throws a divide by zero exception (look at
            #   the gaussian pdf formula).
            p_cond = norm.pdf(data, mu, np.where(std > 0, std, eps))
            # We take the logarithm of probabilities - we have to take care of undefined
            # values.
            p_cond[:, std == 0] = 0
            p_cond = np.log(np.where(p_cond > 0, p_cond, 1))
            # TODO: Compute the likelihood (decision function) for every image and class.
            class_likelihood[:, class_idx] = None
        
        return class_likelihood
    
    def predict(self, data : np.ndarray) -> np.ndarray:
        """Computes the class prediction for each data sample."""
        # This method should result in the final class prediction which is the one with
        # highest likelihood - the right class is probably the most probable one.
        return np.argmax(self.likelihood(data), axis=1)

def main():
    # Ensure that the data have the same form as in 'lab04a_nb_bin'.
    train = lab04_help.MnistDataset("mnist_train.npz")
    test = lab04_help.MnistDataset("mnist_test.npz")

    # TODO: Implement the 'fit' method of 'GaussianNB' at the top of the source file.
    # - We train the gaussian bayes classifer by estimating means and standard deviations
    #   of each class, which are then used for prediction.
    bayes_classifier = GaussianNB()
    bayes_classifier.fit(train.imgs, train.labels)

    # TODO: Now, complete the body of the function 'GaussianNB.likelihood()' used by
    # 'GaussianNB.predict()' to get the final class predictions.
    train_predictions = bayes_classifier.predict(train.imgs)
    test_predictions = bayes_classifier.predict(test.imgs)

    # TODO: Compute the accuracy of the classification.
    # - How many of the selected class indices match the original labels?
    train_accuracy = None
    test_accuracy = None

    print("Train set accuracy {:.2f}%".format(100 * train_accuracy))
    print("Test set accuracy {:.2f}%".format(100 * test_accuracy))

    # Let's look at images created from the probabilities, they should resemble the digits.
    visualize_gaussian_bayes(bayes_classifier, int(np.round(np.sqrt(train.imgs.shape[1]))))

def visualize_gaussian_bayes(bayes_classifier : GaussianNB, side : int) -> None:
    fig, ax = plt.subplots(
        3, len(bayes_classifier.class_names), figsize=(18, 6), subplot_kw={'aspect': 'equal'}
    )
    for class_idx in range(len(bayes_classifier.class_names)):
        # Visualising random samples
        ax[0, class_idx].set_title(f"Rnd. sample ({bayes_classifier.class_names[class_idx]})")
        ax[0, class_idx].imshow(np.reshape(np.random.normal(
            bayes_classifier.mean_class[:, class_idx],
            bayes_classifier.std_class[:, class_idx]
        ), [side, side]), cmap='gray')
        # Visualising the distribution mean
        ax[1, class_idx].set_title(f"Mean ({bayes_classifier.class_names[class_idx]})")
        ax[1, class_idx].imshow(np.reshape(
            bayes_classifier.mean_class[:, class_idx], [side, side]
        ), cmap='gray')
        # Visualising the distribution standard deviation
        ax[2, class_idx].set_title(f"Std. ({bayes_classifier.class_names[class_idx]})")
        ax[2, class_idx].imshow(np.reshape(
            bayes_classifier.std_class[:, class_idx], [side, side]
        ), cmap='gray')
    fig.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
