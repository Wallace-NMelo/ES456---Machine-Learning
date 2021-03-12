from pathlib import Path
import matplotlib.pyplot as plt

from knn import KNN
from aux_methods import accuracy, mean_squared_error, train_test_split
from dataset import wine_data, current_path

X, Y = wine_data.iloc[:, 1:].astype(float).values, wine_data.iloc[:, 0].astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)


k_accuracy = dict()
knn_start, knn_end = 2, 30
for _knn in range(knn_start, knn_end):
    clf = KNN(k=_knn)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    _accuracy = accuracy(y_test, predictions)
    k_accuracy[_knn] = _accuracy
    print("Knn = {0}, Accuracy = {1}%".format(_knn, _accuracy))

accuracy_max = max(k_accuracy.values())
k_max = [k for k in k_accuracy if k_accuracy[k] == accuracy_max]
print("Maximal Value : K = {0}, Accuracy = {1}%".format(k_max, accuracy_max))


plt.plot(list(range(knn_start, knn_end)), list(k_accuracy.values()))
plt.ylabel('Accuracy(%)')
plt.xlabel('k_value')
plt.savefig(current_path.joinpath('q2_knn.png'))
plt.close()