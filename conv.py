import numpy as np
from scipy import signal

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
k = np.array([[1,2],[3,4]])

x = np.array([[a[0,0], a[0,1], a[1,0], a[1,1]], [a[0,1], a[0,2], a[1,1], a[1,2]], [a[1,0], a[1,1], a[2,0], a[2,1]], [a[1,1], a[1,2], a[2,1], a[2,2]]])
x_dash = np.linalg.pinv(x)
print(x_dash)

y = signal.convolve2d(a, np.rot90(k, 2), 'valid')
l = y.reshape((1,4))
print(np.dot(l, x_dash))

print("Original kernel: ", k.reshape((1,4)))
print("Retrieved kernel: ", np.dot(l, x_dash))
