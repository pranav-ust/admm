from sklearn.datasets import load_digits
import numpy as np
# return dictionary digits where digits['data'] means 1797*64 input and digits['target'] denotes the corresponding binary class
# tpye(digits['data']) = numpy.array, type(digits.target) = numpy.array
# digits['images] denotes 1797*(8*8) images
def load_MINISTDATA_binary():
    digits = {}
    sam = load_digits()
    digits['images'] = sam.images
    digits['data'] = sam.data
    digits['target'] = np.array([1 if float(item) % float(2) == 0 else -1 for item in sam.target])
    return digits

digits = load_MINISTDATA_binary()
print digits.keys()
#print digits['target']
print digits['data'].shape
#print digits['images']