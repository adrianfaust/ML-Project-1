import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from src.adeline import adeline
from src.perceptron import Perceptron
from src.perceptron_absorbed_bias import PerceptronAbsorbedBias

df = pd.read_csv(
    '../iris.data',
    header=None, encoding='utf-8')
# select setosa and versicolor
y = df.iloc[0:100, 4].values

y = np.where(y == 'Iris-setosa', 0, 1)
# extract sepal length and petal length

X = df.iloc[0:100, [0, 1, 2, 3]].values
print(X)

# plot data

plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='Setosa')

plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='s', label='Versicolor')

plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')

plt.legend(loc='upper left')
plt.show()

ppn = Perceptron()
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Perceptron')
plt.ylim(bottom=0)
plt.show()

ppn_absorbed = PerceptronAbsorbedBias()
ppn_absorbed.fit(X, y)
plt.plot(range(1, len(ppn_absorbed.errors_) + 1), ppn_absorbed.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Perceptron Absorbed Bias')
plt.ylim(bottom=0)
plt.show()

ppa = adeline()
ppa.fit(X, y)
plt.plot(range(1, len(ppa.losses_) + 1), ppa.losses_, marker='x')
plt.xlabel('Epochs')
plt.ylabel('Mean squared Error')
plt.title('Adeline')
plt.ylim(bottom=0)
plt.show()


def plot_decision_regions_3d(X, y, classifier, resolution=0.02, show_after=False):
    """Plots the decision regions for 3D data.

    Args:
        X: Feature matrix of shape (n_samples, 3).
        y: Target vector of shape (n_samples,).
        classifier: Trained classifier object.
        resolution: Resolution of the meshgrid.
        show_after: Whether to immediately show the plot.
    """

    markers = ('o', 's', '^', 'v', '<')  # Add more if needed
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Create meshgrid for 3D
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x3_min, x3_max = X[:, 2].min() - 1, X[:, 2].max() + 1  # Add z-axis limits

    xx1, xx2, xx3 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                np.arange(x2_min, x2_max, resolution),
                                np.arange(x3_min, x3_max, resolution))

    # Predict on the meshgrid
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel(), xx3.ravel()]).T)  # Include z
    lab = lab.reshape(xx1.shape)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the decision surface (using a scatter plot for 3D)
    ax.scatter(xx1.ravel(), xx2.ravel(), xx3.ravel(), c=lab.ravel(), alpha=0.2, cmap=cmap)  # Alpha for transparency

    # Plot the data points
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], X[y == cl, 2],
                   alpha=0.8, c=colors[idx], marker=markers[idx], label=f'Class {cl}', edgecolor='black')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')  # Set z-axis label

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_zlim(x3_min, x3_max)

    ax.legend()

    if show_after:
        plt.show()

    return fig, ax

# plot_decision_regions(X, y, ppn, show_after=True)
# plot_decision_regions(X, y, ppa, show_after=True)
