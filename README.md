#Implementation of ADMM for neural networks

In this project we have implemented ADMM optimization methods for the convolution layers.
For the convolution layers, we considered converting a convoluted matrix into a dense toeplitz matrix.
The weights updates of kernels are done by solving a system of equations by sampling equations obtained from Toeplitz matrix.
We found that our method worked faster than author's implementation since they only considered the pseudoinverse of the matrix.
We used MNIST digit dataset (odd-even classification) and found that our method gives 89.6% over traditional backprop method 79.6%.
