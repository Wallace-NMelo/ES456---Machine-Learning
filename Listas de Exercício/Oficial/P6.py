from aux_methods import mean, cov_matrix, label_prob
from dataset import iris_data

X, Y = iris_data.drop(['Class'], axis=1).values, iris_data['Class']

print("Iris dataset mean = {}\n".format(mean(X)))

print("iris dataset covariance matrix =\n {}\n".format(cov_matrix(X)))
labels, label_prob = label_prob(Y)
print("Label probability: \n{0} \n{1}".format(labels, label_prob))
