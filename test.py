import mnist_loader
import numpy as np
(train_data_X, train_data_Y), v, (tx, ty) = mnist_loader.load_data('./data/mnist.pkl.gz')
train_data_Y = one_hot_10(train_data_Y, size=10)
ty = one_hot_10(ty, size=10)
train_data_X = np.reshape(train_data_X, [-1, 28, 28, 1])
tx = np.reshape(tx, [-1, 28, 28, 1])

print(train_data_X)
