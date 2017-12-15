import numpy as np
from sklearn.datasets import load_digits
from numpy import vectorize

class ADMM:
    def __init__(self):
        #load the MNIST data
        self.digits = load_digits()
        self.feat_num = 64
        self.layer_1_units = 10
        self.layer_2_units = 5
        self.beta = 10
        self.gamma = 1
        self.grow_rate = 5
        self.warm_start = 10
        self.err_tol = 0.0001
        #data loader, train data and test data
        self.train_data_x = np.transpose(self.digits.data[:1000])
        self.train_data_y = self.convert_binary(0, 1000)
        self.test_data_x = np.transpose(self.digits.data[:1000])
        self.test_data_y = self.convert_binary(0, 1000)
        #related variables
        print("Outputs are ", self.train_data_x)
        self.data_num = self.train_data_y.size
        print(self.train_data_x.shape)
        self.a_0 = self.train_data_x
        self.a_0_pinv = np.linalg.pinv(self.a_0)
        self.W_1 = np.zeros((self.layer_1_units, self.feat_num))
        self.init_var = 1
        self.z_1 = self.init_var * np.random.randn(self.layer_1_units, self.data_num)  # initialize the weights
        self.a_1 = self.init_var * np.random.randn(self.layer_1_units, self.data_num)

        self.W_2 = np.zeros((self.layer_2_units, self.layer_1_units))
        self.z_2 = self.init_var * np.random.randn(self.layer_2_units, self.data_num)
        self.a_2 = self.init_var * np.random.randn(self.layer_1_units, self.data_num)

        self.W_3 = np.zeros((1, self.layer_2_units))
        self.z_3 = self.init_var * np.random.randn(1, self.data_num)

        self._lambda = np.zeros((1, self.data_num))
        #--------------------------------
        self.vactivation = vectorize(self.activation)
        self.vget_z_l = vectorize(self.get_z_l)
        self.vget_z_L = vectorize(self.get_z_L)
        self.vget_predict = vectorize(self.get_predict)
        self.vget_loss = vectorize(self.get_loss)
        #----train and test the deep learning model----------------
        self.train()
        self.test()
    #convert the data into binary classified
    def convert_binary(self, m, n):
        digit = load_digits()
        targets = digit.target[m:n]
        for i in range(n - m):
            if (targets[i] % 2) == 0:  # convert the target into 1 & -1
                targets[i] = 1
            else:
                targets[i] = -1
        return targets

    def get_z_l(self, a, w_a):
        def f_z(z):
            return self.gamma * (a - self.activation(z)) ** 2 + self.beta * (z - w_a) ** 2

        z1 = max((a * self.gamma + w_a * self.beta) / (self.beta + self.gamma), 0)
        result1 = f_z(z1)

        z2 = min(w_a, 0)
        result2 = f_z(z2)

        if result1 <= result2:
            return z1
        else:
            return z2

    def get_z_L(self, y, w_a, _lambda):
        if y == -1:
            def f_z(z):
                return self.beta * z ** 2 - (2 * self.beta * w_a - _lambda) * z + max(1 + z, 0)

            z1 = min((2 * self.beta * w_a - _lambda) / (2 * self.beta), -1)
            z2 = max((2 * self.beta * w_a - _lambda - 1) / (2 * self.beta), -1)
            if f_z(z1) < f_z(z2):
                return z1
            else:
                return z2

        if y == 1:
            def f_z(z):
                return self.beta * z ** 2 - (2 * self.beta * w_a - _lambda) * z + max(1 - z, 0)

            z1 = min((2 * self.beta * w_a - _lambda + 1) / (2 * self.beta), 1)
            z2 = max((2 * self.beta * w_a - _lambda) / (2 * self.beta), 1)

            if f_z(z1) < f_z(z2):
                return z1
            else:
                return z2

        else:
            print("error class: {}".format(y))
            exit()
    # Relu activation function
    def activation(self, i):  # Relu activation function
        if i > 0:
            return i
        else:
            return 0

    def get_predict(self,pre):
        if pre >= 0:
            return 1
        else:
            return -1

    def get_loss(self,pre, gt):
        if gt == -1:
            return max(1 + pre, 0)
        elif gt == 1:
            return max(1 - pre, 0)
        else:
            print("invalid gt..")
            exit()
    # back propagation
    def update(self, is_warm_start=False):
        #global z_1, z_2, z_3, _lambda, W_1, W_2, W_3
        # update layer 1
        old_W_1 = self.W_1
        old_z_1 = self.z_1
        self.W_1 = np.dot(self.z_1, self.a_0_pinv)
        a_1_left = np.linalg.inv((self.beta * np.dot(np.transpose(self.W_2), self.W_2) + self.gamma * np.eye(self.layer_1_units, dtype=float)))
        a_1_right = (self.beta * np.dot(np.transpose(self.W_2), self.z_2) + self.gamma * self.vactivation(self.z_1))
        self.a_1 = np.dot(a_1_left, a_1_right)
        self.z_1 = self.vget_z_l(self.a_1, np.dot(self.W_1, self.a_0))

        # update layer 2
        self.W_2 = np.dot(self.z_2, np.linalg.pinv(self.a_1))
        # numpy.linalg.linalg.LinAlgError: Singular matrix
        a_2_left = np.linalg.inv((self.beta * np.dot(np.transpose(self.W_3), self.W_3) + self.gamma * np.eye(self.layer_2_units, dtype=float)))
        a_2_right = (self.beta * np.dot(np.transpose(self.W_3), self.z_3) + self.gamma * self.vactivation(self.z_2))
        self.a_2 = np.dot(a_2_left, a_2_right)
        self.z_2 = self.vget_z_l(self.a_2, np.dot(self.W_2, self.a_1))

        # update last layer
        self.W_3 = np.dot(self.z_3, np.linalg.pinv(self.a_2))
        self.z_3 = self.vget_z_L(self.train_data_y, np.dot(self.W_3, self.a_2), self._lambda)

        # print("z_3: ")
        # print(z_3)
        loss = self.vget_loss(self.z_3, self.train_data_y)
        # print("loss: {}".format(loss))
        # print("lambda: ")
        # print(_lambda)


        if not is_warm_start:
            self._lambda = self._lambda + self.beta * (self.z_3 - np.dot(self.W_3, self.a_2))

        # ret = np.linalg.norm(z_3 - np.dot(W_3,a_2),2)
        # ret = np.linalg.norm(old_W_1-W_1,2)
        ret = np.linalg.norm(old_z_1 - self.z_1, 2)
        return ret

    def test(self):
        a_0 = self.test_data_x
        layer_1_output = self.vactivation(np.dot(self.W_1, a_0))
        layer_2_output = self.vactivation(np.dot(self.W_2, layer_1_output))
        predict = np.dot(self.W_3, layer_2_output)
        pre = self.vget_predict(predict)

        print("layer 1 value: \n")
        print(layer_1_output)
        print("layer 2 value: \n")
        print(layer_2_output)
        print("layer 3 value: \n")
        print(predict)
        hit = np.equal(pre, self.test_data_y)
        acc = np.sum(hit) / 1000
        print("test data predict accuracy: {}".format(acc))

    def train(self):
        global beta, gamma
        # warm start
        for i in range(self.warm_start):
            loss = self.update(is_warm_start=True)
            print("warm start, err :{}".format(loss))
        # real start
        i = 1
        while 1:
            loss = self.update(is_warm_start=False)
            print("iteration {}, err :{}".format(i, loss))
            # if i%100 == 0:
            # 	beta =  grow_rate* beta
            # 	gamma = gamma* beta

            if i % 20 == 0:
                self.test()
            i = i + 1
            if loss < self.err_tol:
                break


ob = ADMM()