import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.perceptron_absorbed_bias import PerceptronAbsorbedBias

df = pd.read_csv(
    '../iris.data',
    header=None, encoding='utf-8')
y = df.iloc[0:150, 4].values
X = df.iloc[0:150, [0, 1, 2 ,3]].values


def train(classifier, X, y, plot=True, title="Training"):
    classifier.fit(X, y)
    if plot:
        plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
        plt.title(title)
        plt.ylim(bottom=0)
        plt.show()

# NOTE: Switched classifications to 1/0 for easier readability.
setosa_classifier = PerceptronAbsorbedBias()
setosa_labels = np.where(y == 'Iris-setosa', 1, 0)
# print(setosa_labels)
train(setosa_classifier, X, setosa_labels, title="Setosa vs. All Classifier")

versicolor_classifier = PerceptronAbsorbedBias()
versicolor_labels = np.where(y[0:150] == 'Iris-versicolor', 1, 0)
# print(versicolor_labels)
train(versicolor_classifier, X, versicolor_labels, title="Versicolor vs. All Classifier")

virginica_classifier = PerceptronAbsorbedBias()
virginica_labels = np.where(y[0:150] == 'Iris-virginica', 1, 0)
# print(virginica_labels)
train(virginica_classifier, X, virginica_labels, title="Virginica vs. All Classifier")

'''
NOTE: As we can see Setosa can be classified well, but not virginica/versicolor.
The virginica_classifier performs slightly better so we will use that.
'''

def multi_perceptron_classify(X, setosa_classifier, virginica_classifier):
    n = X.shape[0]
    labels = [""] * n

    # "Iris-setosa" or not prediction set.
    setosa_predictions = setosa_classifier.predict(X)
    for i in range(n):
      if setosa_predictions[i] == 1:
        labels[i] = "Iris-setosa"

    # Filter previously classified samples
    remaining_indices = np.where(setosa_predictions == 0)[0]
    X_remaining = X[remaining_indices]

    # Predict remaining samples
    if len(X_remaining) > 0:
        virginica_predictions = virginica_classifier.predict(X_remaining)

        for i, original_index in enumerate(remaining_indices):
            if virginica_predictions[i] == 1:
                labels[original_index] = "Iris-virginica"
            else:
                labels[original_index] = "Iris-versicolor"

    return labels

def get_prediction_accuracy(predicted_labels, ground_truth, print_misclassified=False):
    comparison_results = []
    correct_count = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] == predicted_labels[i]:
            correct_count += 1
        else:
            if print_misclassified:
                print(f"predicted_labels[{i}] : {predicted_labels[i]}, truth: {ground_truth[i]}")

    return correct_count / len(ground_truth)
predicted_labels =multi_perceptron_classify(X, setosa_classifier, virginica_classifier)
print(f"Accuracy of Multi-Perceptron Classification: {get_prediction_accuracy(predicted_labels, y)}")
