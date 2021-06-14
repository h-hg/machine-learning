import numpy as np

class Perceptron:
    def __init__(self, max_iter, lr=0.001):
        self.lr = lr
        self.w = None
        self.b = None
        self.max_iter = max_iter

    def fit(self, X, y):
        # init the parameter
        alpha = np.zeros((X.shape[0], 1))
        # Gram matrix
        G = X @ X.T
        for _ in range(self.max_iter):
            z = self.lr * (alpha * y).reshape((1, -1)) @ (G + 1)
            idx = y * z <= 0
            # update
            alpha[idx] += 1
        # compute
        self.w = np.sum((self.lr * alpha * y).reshape((-1,1)) @ X, axis=0).reshape((-1,1))
        self.b = alpha @ y

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

if __name__ == "__main__":
    row_v = np.array([1, 2, 3])
    row_m = np.array([1,2, 3]).reshape(1, -1)
    col = np.array([1, 2, 3]).reshape(-1,1)
    print(row_v.shape, col.shape)
    print(row_v @ col)
    print(row_m.shape, col.shape)
    print(row_m @ col)
    print((row_v * row_m).shape)
