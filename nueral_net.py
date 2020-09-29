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
        
