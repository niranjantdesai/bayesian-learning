import math

import numpy as np

from pbp_code.PBP_net.PBP_net import PBPNet

np.random.seed(1)
# sys.path.append('PBP_net/')

# We load the boston housing dataset

data = np.loadtxt('../data/boston_housing.txt')

# We obtain the features and the targets

X = data[:, :(data.shape[1]-1)]
y = data[:, data.shape[1]-1]

# We create the train and test sets with 90% and 10% of the data

permutation = np.random.choice(X.shape[0],
                               X.shape[0], replace=False)

size_train = int(np.round(X.shape[0] * 0.9).item())
print(size_train)
index_train = permutation[:size_train]
index_test = permutation[size_train:]

X_train = X[index_train, :]
y_train = y[index_train]
X_test = X[index_test, :]
y_test = y[index_test]

# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.

n_hidden_units = 50
net = PBPNet(X_train, y_train,
             [n_hidden_units, n_hidden_units], normalize=True, n_epochs=40)

# We make predictions for the test set

m, v, v_noise = net.predict(X_test)

# We compute the test RMSE

rmse = np.sqrt(np.mean((y_test - m)**2))

print('Test RMSE: ', rmse)

# We compute the test log-likelihood

test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) -
                  0.5 * (y_test - m)**2 / (v + v_noise))

print('Test LL: ', test_ll)