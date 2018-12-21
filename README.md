# Parallelized Neural Networks Optimization without Backpropagation

In this project we have implemented ADMM optimization methods for the convolution layers.
For the convolution layers, we considered converting a convoluted matrix into a dense toeplitz matrix.
The weights updates of kernels are done by solving a system of equations by sampling equations obtained from Toeplitz matrix.
We found that our method worked faster than author's implementation since they only considered the pseudoinverse of the matrix.
We used MNIST digit dataset (odd-even classification) and found that our method gives 89.6% over traditional backprop method 79.6% *(we repeat, it's odd-even classification, not digit classification problem. It's slightly harder to get better accuracy on odd-even as compared to digit classification)*.


## Requirements

You need Python 3, `sklearn`, and `scipy` for this.

## Report

Report is [here](https://github.com/pranav-ust/admm/blob/master/report/report.pdf).

## Usage

For ADMM based optimization, run `python3 admm.py`

For baseline, run `python3 baseline.py`
