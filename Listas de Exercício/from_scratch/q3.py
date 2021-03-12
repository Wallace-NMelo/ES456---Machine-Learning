import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN


# Loading dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(type(X_train), type(X_test))


# Test
for _knn in range(2,9):
    clf = KNN(k=_knn)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    accuracy = np.sum(predictions == y_test)/len(y_test)
    print("Knn = {0}, Accuracy = {1}".format(_knn, accuracy))