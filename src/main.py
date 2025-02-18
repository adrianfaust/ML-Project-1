import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from src.adeline import adeline
from src.perceptron import Perceptron
from src.perceptron_absorbed_bias import PerceptronAbsorbedBias
from src.AdelineSGD import AdalineSGD

df = pd.read_csv(
    '../iris.data',
    header=None, encoding='utf-8')
# select setosa and versicolor
y = df.iloc[0:100, 4].values

y = np.where(y == 'Iris-setosa', 0, 1)
# extract sepal length and petal length

X = df.iloc[0:100, [0,2]].values
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

pps = AdalineSGD();
pps.fit_mini_batch_SGD(X, y)
#pps.fit(X, y)
plt.plot(range(1, len(pps.losses_) + 1), pps.losses_,
            marker='o')
print(pps.losses_)
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()
plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02, show_after=False):
    # setup marker generator and color map
    markers = ('o', 's',
    '^'
    , 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}'
                    ,
                    edgecolor='black')

plot_decision_regions(X, y, ppn)
plt.title('Perceptron Decision Boundary')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.savefig("../figs/perceptron_decision_boundary.eps")
plt.show()
plot_decision_regions(X, y, ppa, show_after=True)
plt.title('Adeline Decision Boundary')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.savefig("../figs/adeline_decision_boundary.eps")
plt.show()
