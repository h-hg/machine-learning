import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import sklearn

class LinearRegression:
    def __init__(self, batch_size=100, epoch = 1000, lr = 0.05):
        # hyperparameter
        self.batch_size = batch_size
        self.epoch = epoch # iteration number
        self.lr = lr # learning rate

    def forward(self, X):
        return X @ self.w + self.b
    
    # loss function
    def loss(self, X, y):
        return np.mean((self.forward(X) - y) ** 2)

    def back_grad(self, X, y):
        z = self.forward(X)
        grad_w = np.mean((z - y) * X, axis=0).reshape((-1, 1))
        grad_b = np.mean((z - y), axis=0)
        return grad_w, grad_b

    def update(self, grad):
        grad_w, grad_b = grad
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b

    def data_iter(self, X, y):
        m = X.shape[0]
        indices = np.arange(m)
        np.random.shuffle(indices)
        for i in range(0, m, self.batch_size):
            j = indices[i:min(i + self.batch_size, m)]
            yield X.take(j, axis=0), y.take(j, axis=0)

    def train(self, X, y):
        for _ in range(self.epoch):
            for X_batch, y_batch in self.data_iter(X, y):
                grad = self.back_grad(X_batch, y_batch)
                self.update(grad)
            self.record_loss(X, y)

    def fit(self, X_train, y_train, X_test, y_test):
        y_train = y_train.reshape((-1, 1))
        n = X_train.shape[1] # parameter number
        self.w = np.zeros((n, 1))
        self.b = 0

        self.X_test = X_test
        self.y_test = y_test.reshape((-1, 1))
        # variable for plot
        self.losses_train = []
        self.losses_test = []

        self.train(X_train, y_train)

    def record_loss(self, x,y):
        self.losses_train.append(self.loss(x,y))
        self.losses_test.append(self.loss(self.X_test,self.y_test))


    def plot_loss(self):
        plt.figure()
        plt.plot(range(len(self.losses_train)), self.losses_train, color = 'red',label='Train')
        plt.plot(range(len(self.losses_test)), self.losses_test, color = 'blue',label='Test')
        plt.title("Convergence Graph of Loss Function")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')
        plt.show()

def load_data():
    X,y = sklearn.datasets.load_boston(return_X_y=True)
    y = y.reshape((-1, 1))
    def normalize(dataset):
        mu = np.mean(dataset, axis=0)
        sigma = np.std(dataset, axis=0)
        return (dataset - mu) / sigma
    X = normalize(X)
    return X, y

if __name__ == "__main__":

    X,y = load_data()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
    model = LinearRegression()
    model.fit(X_train, y_train, X_test, y_test)
    # print(model.losses_train)
    # print(model.losses_test)
    model.plot_loss()
