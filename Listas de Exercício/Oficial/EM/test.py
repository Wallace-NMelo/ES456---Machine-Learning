import numpy as np
from GMM import GMM
from scipy.stats import mode
from aux_methods import accuracy
from sklearn.datasets import load_iris, load_wine


wine = load_wine()
clusters_wine = len(np.unique(wine.target))

gmm = GMM(clusters=clusters_wine, max_iter=10)
gmm.fit(wine.data)

permutation = np.array([mode(wine.target[gmm.predict(wine.data) == i]).mode.item() for i in range(gmm.clusters)])
wine_pred = permutation[gmm.predict(wine.data)]

iris = load_iris()
clusters_iris = len(np.unique(iris.target))
np.random.seed(42)
gmm_ = GMM(clusters=clusters_iris, max_iter=10)
gmm_.fit(iris.data)

permutation = np.array([mode(iris.target[gmm_.predict(iris.data) == i]).mode.item() for i in range(gmm_.clusters)])
iris_pred = permutation[gmm_.predict(iris.data)]

print("GMM:\n")
print("Accuracy_wine = {}".format(accuracy(wine_pred, wine.target)))
print("Accuracy_iris = {}".format(accuracy(iris_pred, iris.target)))
