import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None

    def fix(self, X, y):
        y = y.reshape((-1, 1))
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.w

