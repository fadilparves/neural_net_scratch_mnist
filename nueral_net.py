import numpy as np
import math
import struct

class NNClassificationModel:

    def __init__(self, n_classes, n_features, n_hidden_units=30, l1=0.0, l2=0.0, epochs=1000, learning_rate=0.01, n_batches=1, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.w1, self.w2 = self._init_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_batches = n_batches

    def _init_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=(self.n_hidden_units, self.n_features))
        w2 = np.random.uniform(-1.0, 1.0, size=(self.n_classes, self.n_hidden_units))

        return w1, w2
    
    #Training part
    def fit(self, X, y):
        self.error_ = []
        X_data, y_data = X.copy(), y.copy()
        y_data_enc = one_hot(y_data, self.n_classes)

        X_mbs = np.array_split(X_data, self.n_batches)
        y_mbs = np.array_split(y_data_enc, self.n_batches)

        for i in range(self.epochs):
            epoch_errors = []

            for Xi, yi in zip(X_mbs, y_mbs):
                #Update the weights
                error, grad1, grad2 = self._backprop_step(Xi, yi)
                epoch_errors.append(error)
                self.w1 -= self.learning_rate * grad1
                self.w2 -= self.learning_rate * grad2
            
            self.error_.append(np.mean(epoch_errors))
        
        return self

    def _backprop_step(self, X, y):
        net_input, net_hidden, act_hidden, net_out, act_out = self._forward(X)
        y = y.T

        grad1 , grad2 = self._backward(net_input, net_hidden, act_hidden, act_out, y)

        grad1 += self.w1 * (self.l1 + self.l2)
        grad2 += self.w2 * (self.l1 + self.l2)

        error = self._error(y, act_out)

        return error, grad1, grad2

    def _forward(self, X):
        net_input = X.copy()
        net_hidden = self.w1.dot(net_input.T)
        act_hidden = self.sigmoid(net_hidden)
        net_out = self.w2.dot(act_hidden)
        act_out = self.sigmoid(net_out)
        return net_input, net_hidden, act_hidden, net_out, act_out

    def _backward(self, net_input, net_hidden, act_hidden, act_out, y):
        sigma3 = act_out - y
        sigma2 = self.w2.T.dot(sigma3) * self.sigmoid_prime(net_hidden)
        grad1 = sigma2.dot(net_input)
        grad2 = sigma3.dot(act_hidden.T)
        return grad1, grad2

    def sigmoid(self, inp):
        return 1.0 / (1.0 + np.exp(-inp))

    def sigmoid_prime(self, inp):
        sg = self.sigmoid(inp)
        return sg * (1 - sg)

    def cross_entropy(self, outputs, y_target):
        return -np.sum(np.log(outputs) * y_target, axis=1)

    def _error(self, y, output):
        L1_term = self.L1_reg(self.l1, self.w1, self.w2)
        L2_term = self.L2_reg(self.l2, self.w1, self.w2)
        error = self.cross_entropy(output, y) + L1_term + L2_term
        retrun 0.5 * np.mean(error)

    def L1_reg(self, lambda_, w1, w2):
        return(lambda_ / 2.0) * (np.sum(w1 ** 2) + np.sum(w2 ** 2))

    def L2_reg(self, lambda_, w1, w2):
        return(lambda_ / 2.0) * (np.abs(w2).sum() + np.abs(w2).sum())

    def predict(self, X):
        Xt = X.copy()
        _, _, _, net_out, _ = self._forward(Xt)
        return self.mle(net_out.T)

    def mle(self, y, axis=1):
        return np.argmax(y, axis)