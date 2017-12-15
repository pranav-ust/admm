# coding=utf-8

import numpy
import struct
import matplotlib.pyplot as plt
import math
import random
import copy
# test
#from BasicMultilayerNeuralNetwork import BMNN2


def sigmoid(inX):
    if 1.0 + numpy.exp(-inX) == 0.0:
        return 999999999.999999999
    return 1.0 / (1.0 + numpy.exp(-inX))


def difsigmoid(inX):
    return sigmoid(inX) * (1.0 - sigmoid(inX))


def tangenth(inX):
    return (1.0 * math.exp(inX) - 1.0 * math.exp(-inX)) / (1.0 * math.exp(inX) + 1.0 * math.exp(-inX))


def cnn_conv(in_image, filter_map, B, type_func='sigmoid'):
    # in_image[num,feature map,row,col]=>in_image[Irow,Icol]
    # features map[k filter,row,col]
    # type_func['sigmoid','tangenth']
    # out_feature[k filter,Irow-row+1,Icol-col+1]
    shape_image = numpy.shape(in_image)  # [row,col]
    # print "shape_image",shape_image
    shape_filter = numpy.shape(filter_map)  # [k filter,row,col]
    if shape_filter[1] > shape_image[0] or shape_filter[2] > shape_image[1]:
        raise Exception
    shape_out = (shape_filter[0], shape_image[0] - shape_filter[1] + 1, shape_image[1] - shape_filter[2] + 1)
    out_feature = numpy.zeros(shape_out)
    k, m, n = numpy.shape(out_feature)
    for k_idx in range(0, k):
        # rotate 180 to calculate conv
        c_filter = numpy.rot90(filter_map[k_idx, :, :], 2)
        for r_idx in range(0, m):
            for c_idx in range(0, n):
                # conv_temp=numpy.zeros((shape_filter[1],shape_filter[2]))
                conv_temp = numpy.dot(in_image[r_idx:r_idx + shape_filter[1], c_idx:c_idx + shape_filter[2]], c_filter)
                sum_temp = numpy.sum(conv_temp)
                if type_func == 'sigmoid':
                    out_feature[k_idx, r_idx, c_idx] = sigmoid(sum_temp + B[k_idx])
                elif type_func == 'tangenth':
                    out_feature[k_idx, r_idx, c_idx] = tangenth(sum_temp + B[k_idx])
                else:
                    raise Exception
    return out_feature


def cnn_maxpooling(out_feature, pooling_size=2, type_pooling="max"):
    k, row, col = numpy.shape(out_feature)
    max_index_Matirx = numpy.zeros((k, row, col))
    out_row = int(numpy.floor(row / pooling_size))
    out_col = int(numpy.floor(col / pooling_size))
    out_pooling = numpy.zeros((k, out_row, out_col))
    for k_idx in range(0, k):
        for r_idx in range(0, out_row):
            for c_idx in range(0, out_col):
                temp_matrix = out_feature[k_idx, pooling_size * r_idx:pooling_size * r_idx + pooling_size,
                              pooling_size * c_idx:pooling_size * c_idx + pooling_size]
                out_pooling[k_idx, r_idx, c_idx] = numpy.amax(temp_matrix)
                max_index = numpy.argmax(temp_matrix)
                # print max_index
                # print max_index/pooling_size,max_index%pooling_size
                max_index_Matirx[
                    k_idx, pooling_size * r_idx + max_index / pooling_size, pooling_size * c_idx + max_index % pooling_size] = 1
    return out_pooling, max_index_Matirx


def poolwithfunc(in_pooling, W, B, type_func='sigmoid'):
    k, row, col = numpy.shape(in_pooling)
    out_pooling = numpy.zeros((k, row, col))
    for k_idx in range(0, k):
        for r_idx in range(0, row):
            for c_idx in range(0, col):
                out_pooling[k_idx, r_idx, c_idx] = sigmoid(W[k_idx] * in_pooling[k_idx, r_idx, c_idx] + B[k_idx])
    return out_pooling


# out_feature is the out put of conv
def backErrorfromPoolToConv(theta, max_index_Matirx, out_feature, pooling_size=2):
    k1, row, col = numpy.shape(out_feature)
    error_conv = numpy.zeros((k1, row, col))
    k2, theta_row, theta_col = numpy.shape(theta)
    if k1 != k2:
        raise Exception
    for idx_k in range(0, k1):
        for idx_row in range(0, row):
            for idx_col in range(0, col):
                error_conv[idx_k, idx_row, idx_col] = \
                    max_index_Matirx[idx_k, idx_row, idx_col] * \
                    float(theta[idx_k, idx_row / pooling_size, idx_col / pooling_size]) * \
                    difsigmoid(out_feature[idx_k, idx_row, idx_col])
    return error_conv


def backErrorfromConvToInput(theta, inputImage):
    k1, row, col = numpy.shape(theta)
    # print "theta",k1,row,col
    i_row, i_col = numpy.shape(inputImage)
    if row > i_row or col > i_col:
        raise Exception
    filter_row = i_row - row + 1
    filter_col = i_col - col + 1
    detaW = numpy.zeros((k1, filter_row, filter_col))
    # the same with conv valid in matlab
    for k_idx in range(0, k1):
        for idx_row in range(0, filter_row):
            for idx_col in range(0, filter_col):
                subInputMatrix = inputImage[idx_row:idx_row + row, idx_col:idx_col + col]
                # print "subInputMatrix",numpy.shape(subInputMatrix)
                # rotate theta 180
                # print numpy.shape(theta)
                theta_rotate = numpy.rot90(theta[k_idx, :, :], 2)
                # print "theta_rotate",theta_rotate
                dotMatrix = numpy.dot(subInputMatrix, theta_rotate)
                detaW[k_idx, idx_row, idx_col] = numpy.sum(dotMatrix)
    detaB = numpy.zeros((k1, 1))
    for k_idx in range(0, k1):
        detaB[k_idx] = numpy.sum(theta[k_idx, :, :])
    return detaW, detaB


def loadMNISTimage(absFilePathandName, datanum=60000):
    images = open(absFilePathandName, 'rb')
    buf = images.read()
    index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
    print magic, numImages, numRows, numColumns
    index += struct.calcsize('>IIII')
    if magic != 2051:
        raise Exception
    datasize = int(784 * datanum)
    datablock = ">" + str(datasize) + "B"
    # nextmatrix=struct.unpack_from('>47040000B' ,buf, index)
    nextmatrix = struct.unpack_from(datablock, buf, index)
    nextmatrix = numpy.array(nextmatrix) / 255.0
    # nextmatrix=nextmatrix.reshape(numImages,numRows,numColumns)
    # nextmatrix=nextmatrix.reshape(datanum,1,numRows*numColumns)
    nextmatrix = nextmatrix.reshape(datanum, 1, numRows, numColumns)
    return nextmatrix, numImages


def loadMNISTlabels(absFilePathandName, datanum=60000):
    labels = open(absFilePathandName, 'rb')
    buf = labels.read()
    index = 0
    magic, numLabels = struct.unpack_from('>II', buf, index)
    print magic, numLabels
    index += struct.calcsize('>II')
    if magic != 2049:
        raise Exception

    datablock = ">" + str(datanum) + "B"
    # nextmatrix=struct.unpack_from('>60000B' ,buf, index)
    nextmatrix = struct.unpack_from(datablock, buf, index)
    nextmatrix = numpy.array(nextmatrix)
    return nextmatrix, numLabels


def simpleCNN(numofFilter, filter_size, pooling_size=2, maxIter=1000, imageNum=500):
    decayRate = 0.01
    MNISTimage, num1 = loadMNISTimage("train-images-idx3-ubyte", imageNum)
    print num1
    row, col = numpy.shape(MNISTimage[0, 0, :, :])
    out_Di = numofFilter * ((row - filter_size + 1) / pooling_size) * ((col - filter_size + 1) / pooling_size)
    MLP = BMNN2.MuiltilayerANN(1, [128], out_Di, 10, maxIter)
    MLP.setTrainDataNum(imageNum)
    MLP.loadtrainlabel("train-labels-idx1-ubyte")
    MLP.initialweights()
    # MLP.printWeightMatrix()
    rng = numpy.random.RandomState(23455)
    W_shp = (numofFilter, filter_size, filter_size)
    W_bound = numpy.sqrt(numofFilter * filter_size * filter_size)
    W_k = rng.uniform(low=-1.0 / W_bound, high=1.0 / W_bound, size=W_shp)
    B_shp = (numofFilter,)
    B = numpy.asarray(rng.uniform(low=-.5, high=.5, size=B_shp))
    cIter = 0
    while cIter < maxIter:
        cIter += 1
        ImageNum = random.randint(0, imageNum - 1)
        conv_out_map = cnn_conv(MNISTimage[ImageNum, 0, :, :], W_k, B, "sigmoid")
        out_pooling, max_index_Matrix = cnn_maxpooling(conv_out_map, 2, "max")
        pool_shape = numpy.shape(out_pooling)
        MLP_input = out_pooling.reshape(1, 1, out_Di)
        # print numpy.shape(MLP_input)
        DetaW, DetaB, temperror = MLP.backwardPropogation(MLP_input, ImageNum)
        if cIter % 50 == 0:
            print cIter, "Temp error: ", temperror
        # print numpy.shape(MLP.Theta[MLP.Nl-2])
        # print numpy.shape(MLP.Ztemp[0])
        # print numpy.shape(MLP.weightMatrix[0])
        theta_pool = MLP.Theta[MLP.Nl - 2] * MLP.weightMatrix[0].transpose()
        # print numpy.shape(theta_pool)
        # print "theta_pool",theta_pool
        temp = numpy.zeros((1, 1, out_Di))
        temp[0, :, :] = theta_pool
        back_theta_pool = temp.reshape(pool_shape)
        # print "back_theta_pool",numpy.shape(back_theta_pool)
        # print "back_theta_pool",back_theta_pool
        error_conv = backErrorfromPoolToConv(back_theta_pool, max_index_Matrix, conv_out_map, 2)
        # print "error_conv",numpy.shape(error_conv)
        # print error_conv
        conv_DetaW, conv_DetaB = backErrorfromConvToInput(error_conv, MNISTimage[ImageNum, 0, :, :])
        # print "W_k",W_k
        # print "conv_DetaW",conv_DetaW
        # print "conv_DetaB",conv_DetaB
        temp = W_k - decayRate * conv_DetaW
        W_k = copy.deepcopy(temp)
        # print "W_k",W_k
        temp = B - decayRate * conv_DetaB
        B = copy.deepcopy(B)
        # print "B",B
        MLP.updatePara(DetaW, DetaB, 1)
    return W_k, B, MLP


def getTrainAccuracy(numofFilter, filter_size, pooling_size, ImageNum, W_k, B, MLP):
    MNISTimage, num1 = loadMNISTimage("train-images-idx3-ubyte", ImageNum)
    MLP.setTrainDataNum(ImageNum)
    MLP.loadtrainlabel("train-labels-idx1-ubyte")
    # MNISTlabel,num2=loadMNISTimage("F:\Machine Learning\UFLDL\data\common\\train-images-idx3-ubyte",ImageNum)
    row, col = numpy.shape(MNISTimage[0, 0, :, :])
    iteration = 0
    out_Di = numofFilter * ((row - filter_size + 1) / pooling_size) * ((col - filter_size + 1) / pooling_size)
    accuracycount = 0
    while iteration < ImageNum:
        conv_out_map = cnn_conv(MNISTimage[iteration, 0, :, :], W_k, B, "sigmoid")
        out_pooling, max_index_Matrix = cnn_maxpooling(conv_out_map, 2, "max")
        # pool_shape = numpy.shape(out_pooling)
        MLP_input = out_pooling.reshape(1, 1, out_Di)
        Atemp, Ztemp, errorsum = MLP.forwardPropogation(MLP_input, iteration)
        TrainPredict = Atemp[MLP.Nl - 2]
        # print TrainPredict
        Plist = TrainPredict.tolist()
        LabelPredict = Plist[0].index(max(Plist[0]))
        # print "LabelPredict",LabelPredict
        # print "trainLabel",MLP.trainlabel[iteration]
        if int(LabelPredict) == int(MLP.trainlabel[iteration]):
            accuracycount += 1
        iteration += 1
        if iteration % 50 == 0:
            print iteration
    print "accuracy:", float(accuracycount) / float(ImageNum)
    return float(accuracycount) / float(ImageNum)


if __name__ == '__main__':
    MNISTimage, num1 = loadMNISTimage("train-images-idx3-ubyte", 1)
    MNISTlabel, num2 = loadMNISTlabels("train-labels-idx1-ubyte", 1)
    fig1 = plt.figure("convolution")
    k = 10
    filter_size = 5
    rng = numpy.random.RandomState(23455)
    w_shp = (k, filter_size, filter_size)
    w_bound = numpy.sqrt(k * filter_size * filter_size)
    w_k = rng.uniform(low=-1.0 / w_bound, high=1.0 / w_bound, size=w_shp)
    B_shp = (k,)
    B = numpy.asarray(rng.uniform(low=-.5, high=.5, size=B_shp))
    # print B
    out_map = cnn_conv(MNISTimage[0, 0, :, :], w_k, B, "sigmoid")
    for idx in range(0, 10):
        plotwindow = fig1.add_subplot(2, 5, idx + 1)
        plt.imshow(out_map[idx, :, :], cmap='gray')
    # plt.show()
    fig2 = plt.figure("max-pooling")
    out_pooling, max_index = cnn_maxpooling(out_map)
    for idx in range(0, 10):
        plotwindow = fig2.add_subplot(2, 5, idx + 1)
        plt.imshow(out_pooling[idx, :, :], cmap='gray')

    W_pool_shp = (k,)
    W_pool = numpy.asarray(rng.uniform(low=-1, high=1, size=W_pool_shp))
    B_pool_shp = (k,)
    B_pool = numpy.asarray(rng.uniform(low=-.5, high=.5, size=B_pool_shp))
    fig3 = plt.figure("pooling")
    pooling = poolwithfunc(out_pooling, W_pool, B_pool)
    for idx in range(0, 10):
        plotwindow = fig3.add_subplot(2, 5, idx + 1)
        plt.imshow(pooling[idx, :, :], cmap='gray')
    # plt.show()

    W_k, B, MLP = simpleCNN(5, 5, 2, 2000, 10000)
    # MLP.printWeightMatrix()
    accu = getTrainAccuracy(5, 5, 2, 4000, W_k, B, MLP)
    print accu
    pass