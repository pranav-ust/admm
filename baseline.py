from sklearn.datasets import load_digits
import numpy as np

class BaseLine:
    def __init__(self):
        self.digits = self.load_MINISTDATA_binary()
        # input dataset & output dataset
        self.X = self.digits['data']
        self.y = self.digits['target']
        # variables of input and output
        self.num_examples = len(self.X)  # training set size
        self.nn_input_dim = 64  # input layer dimensionality
        self.nn_output_dim = 2  # output layer dimensionality
        # Gradient descent parameters
        self.epsilon = 0.01  # learning rate
        self.reg_lambda = 0.01  # regularization strength
        #----run the model------
        run = self.build_model(3, print_loss=True)

    #load data into binary classes
    def load_MINISTDATA_binary(self):
        digits = {}
        sam = load_digits()
        digits['data'] = sam.data
        digits['target'] = np.array([1 if float(item) % float(2) == 0 else -1 for item in sam.target])
        return digits
    # calculate the loss
    def calculate_loss(self, model):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        z1 = self.X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(self.num_examples), self.y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / self.num_examples * data_loss

    def predict(self, model, x):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def build_model(self, nn_hdim, num_passes=20000, print_loss=False):
        np.random.seed(0)
        W1 = np.random.randn(self.nn_input_dim, nn_hdim) / np.sqrt(self.nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, self.nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, self.nn_output_dim))
        model = {}
        for i in range(0, num_passes):
            #-----Forward propagation-----------
            z1 = self.X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            #------Backpropagation-------------
            delta3 = probs
            delta3[range(self.num_examples), self.y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(self.X.T, delta2)
            db1 = np.sum(delta2, axis=0)
            #----Add regularization terms--------
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1
            #----Gradient descent parameter update----
            W1 += -self.epsilon * dW1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dW2
            b2 += -self.epsilon * db2
            # Set new parameters
            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(model)))

        return model

ob = BaseLine()