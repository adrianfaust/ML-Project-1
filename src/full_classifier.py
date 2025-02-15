import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.perceptron_absorbed_bias import PerceptronAbsorbedBias

df = pd.read_csv(
    '../iris.data',
    header=None, encoding='utf-8')
# select setosa and versicolor
y = df.iloc[0:150, 4].values
print(y)

setosa_labels = np.where(y == 'Iris-setosa', 0, 1)
versicolor_labels = np.where(y[50:150] == 'Iris-versicolor', 0, 1)
virginica_labels = np.where(y[50:150] == 'Iris-virginica', 0, 1)

X = df.iloc[0:150, [0, 1, 2 ,3]].values
no_setosa_X = df.iloc[50:150, [0, 1, 2 ,3]].values
print(no_setosa_X)


def train(classifier, X, y, plot=True, title="Training"):
    classifier.fit(X, y)
    if plot:
        plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
        plt.title(title)
        plt.ylim(bottom=0)
        plt.show()


setosa_classifier = PerceptronAbsorbedBias()
train(setosa_classifier, X, setosa_labels, title="Setosa Classifier")

versicolor_classifier = PerceptronAbsorbedBias()
train(versicolor_classifier, no_setosa_X, versicolor_labels, title="Versicolor Classifier")

virginica_classifier = PerceptronAbsorbedBias()
train(virginica_classifier, no_setosa_X, virginica_labels, title="Virginica Classifier")
classifiers  = {
    "Iris-setosa": setosa_classifier,
    "Iris-versicolor": versicolor_classifier,
    "Iris-virginica": virginica_classifier
}

def classify_with_2(data, setosa_classifier, versicolor_classifier):
    # TODO: Need to call predict on data (list), and replace 1s with the classification, then "and" them together
    # with other classifies lists
    raise NotImplementedError


