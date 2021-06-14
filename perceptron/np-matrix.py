import numpy as np

class Perceptron:
    def __init__(self, max_iter, lr=0.001):
        self.lr = lr
        self.w = None
        self.b = None
        self.max_iter = max_iter

    def fit(self, X, y):
        # init the parameter
        self.w = np.zeros((X.shape[1], 1))
        self.b = 0

        for _ in range(self.max_iter):
            z = X @ self.w + self.b
            # get error set
            error_idx = y * z <= 0
            m = error_idx.size
            X_error = X[error_idx, ...]
            y_error = y[error_idx].reshape((-1,1))
            # update
            self.b -= self.lr * np.sum(- y_error) / m
            self.w -= self.lr * np.sum(- y_error * X_error, axis=0).reshape((-1, 1)) / m

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

if __name__ == "__main__":
    pass