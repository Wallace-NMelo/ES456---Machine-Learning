import numpy as np
from dataset import wine_data
from aux_methods import accuracy, mean_squared_error, train_test_split
from kmeans import KMeans

X, Y = wine_data.iloc[:, 1:].astype(float).values, wine_data.iloc[:, 0].astype(float).values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(np.shape(X))
clusters = len(np.unique(Y))

centroids = [[14.75, 1.73, 2.39, 11.4, 91, 3.1, 3.69, .43, 2.81, 5.4, 1.25, 2.73, 1150],
             [12.7, 3.87, 2.4, 23, 101, 2.83, 2.55, .43, 1.95, 2.57, 1.19, 3.13, 463], \
             [13.73, 4.36, 2.26, 22.5, 88, 1.28, .47, .52, 1.15, 6.62, .78, 1.75, 520]]
k_accuracy = dict()
for k_ in range(2, 10):
    k = KMeans(K=k_, max_iters=50, centroids=centroids, plot_steps=False)
    y_pred = k.predict(X)
    accuracy_ = accuracy(y_pred, Y)
    k_accuracy[accuracy_] = k_
    print("K = {0}, Accuracy = {1}".format(k_, accuracy_))


accuracy_max = max(k_accuracy.keys())
k_max = k_accuracy[accuracy_max]
print("Maximal Value : K = {0}, Accuracy = {1}".format(k_max, accuracy_max))
