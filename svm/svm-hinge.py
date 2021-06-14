import numpy as np

class SVM:
    def __init__(self, C, lr, epoch):
        self.C = C
        self.lr = lr
        self.epoch = epoch
        self.w = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros((n, 1))
        self.b = 0

    def back_grad(self, X, y):
        m, n = X.shape
        z = 1 - y * (X @ self.w + self.b)
        z = z.reshape((-1, 1))
        
        pass