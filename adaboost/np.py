import numpy as np

class BaseLearner:
    def __init__(self):
        pass
    def fit(self, X, y):
        return []
    def predict(self, X, y):
        return []

class AdaBoost:
    def __init__(self, lr, epoch, m):
        self.lr = lr
        self.epoch = epoch
        self.m = m # number of weak model
        self.alpha = None
        self.models = None

    def fit(self, X, y):
        m, n = X.shape
        # a set of weak model
        self.models = []
        # weights of sample
        w = np.full((m, ), 1.0 / m)
        # weights of weak model
        self.alpha = []

        for _ in range(self.m):
            # resample with the weight w
            sample_index = np.random.choice(m, m, p=w)
            X_train = X[sample_index]
            y_train = y[sample_index]

            # build the weak
            weak_model = BaseLearner()
            weak_model.fit(X_train,y_train)
            
            # update 
            y_pred = weak_model.predict(X, y)
            err = np.sum(w[y_pred != y])

            if err > 0.5:
                break

            alpha = np.log((1 - err) / 2) / 2
            tmp = w * np.exp(-alpha * y * y_pred)
            w = tmp / np.sum(tmp)

            self.alpha.append(alpha)
            self.models.append(weak_model)

        self.alpha = np.array(self.alpha).reshape((-1, 1))

    def predict(self, X):
        Y = np.transpose([model.predict(X).reshape(-1) for model in self.models])        
        return np.sign(Y @ self.alpha)
