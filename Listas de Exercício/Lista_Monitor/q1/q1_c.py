import numpy as np

from nb import NaiveBayes

from dataset import data_xor_random
from aux_methods import accuracy, mean_squared_error, train_test_split, higher_dim


# XOR dataset
# A ^ B = C
X_xor, y_xor = data_xor_random[["A", "B"]], np.array(data_xor_random["C"])
# Higher Dimension
X_xor_hdim = higher_dim(X_xor)

X_train_xor, X_test_xor, y_train_xor, y_true_xor = train_test_split(X_xor_hdim, y_xor, test_size=0.2)

nb_xor = NaiveBayes()
nb_xor.fit(np.array(X_train_xor), y_train_xor)
y_predict_nb = nb_xor.predict(np.array(X_test_xor))

print("A ^ B = C")
print("xor dataset Accuracy = {0}% Squared error = {1}%\n".format(accuracy(y_predict_nb, y_true_xor),
                                                                  mean_squared_error(y_predict_nb, y_true_xor)))
