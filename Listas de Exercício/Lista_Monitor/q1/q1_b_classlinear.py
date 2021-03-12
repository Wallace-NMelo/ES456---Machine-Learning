import numpy as np


from nb import NaiveBayes
from dataset import data_xor_random
from aux_methods import accuracy, mean_squared_error, train_test_split


# XOR dataset test

data_X_xor, data_y_xor = np.array(data_xor_random[["A", "B"]]), data_xor_random["C"]
X_train_xor, X_test_xor, y_train_xor, y_true_xor = train_test_split(data_X_xor, data_y_xor, test_size=0.2)

nb_xor = NaiveBayes()
nb_xor.fit(np.array(X_train_xor), y_train_xor)
y_predict_xor = nb_xor.predict(np.array(X_test_xor))


print("xor dataset Accuracy = {0}% Squared error = {1}%\n".format(accuracy(y_predict_xor, y_true_xor),
                                                                mean_squared_error(y_predict_xor, y_true_xor)))

