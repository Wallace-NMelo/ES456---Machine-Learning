import pandas
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from PCA import PCA

iris = datasets.load_iris()
X, y = iris.data, iris.target

print("1) Respostas")
# a)
print("a): {}".format(np.mean(X, axis=0)))
# b)
print("b): {}".format(np.std(X, axis=0)))
# c)
print("c): {}".format(np.amax(X, axis=0)))
# d)
print("d): {}".format(np.amin(X, axis=0)))
# e)

# f)
X_ = X - np.mean(X, axis=0)

print("f): \n {}".format(np.cov(X_.T)))
# g)
print("g): \n {}".format(np.corrcoef(X_.T)))

# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)
print("h) : \n")
print('Shape of X:', X.shape)
print('Shape of transformed X:', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2,
        c=y, edgecolor='none', alpha=0.8,
        cmap=plt.cm.get_cmap('viridis', 3))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()

