import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

if __name__ == "__main__":
    # load data
    X,y = sklearn.datasets.load_boston(return_X_y=True)
    
    # preprocess data
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X - mu) / sigma

    y = y.reshape(-1, 1)
    
    X = torch.tensor(X, dtype=torch.float) # numpy default double
    y = torch.tensor(y, dtype=torch.float) # numpy default double

    # data loader
    ds = torch.utils.data.TensorDataset(X, y)
    n_train = int(len(ds)*0.8)
    n_valid = len(ds) - n_train
    ds_train, ds_valid = torch.utils.data.random_split(ds,[n_train,n_valid])

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=100)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=100)

    # parameters
    w = torch.zeros([X.shape[1], 1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = torch.tensor(0.05)
    losses = []

    # train
    epoch = 100
    for i in range(epoch):
        for features, labels in dl_train:
            # forward
            y_pred = features @ w + b
            loss = torch.mean((y_pred - labels) ** 2)
            
            # backward
            loss.backward()

            with torch.no_grad():
                # update parameters
                w -= lr * w.grad
                b -= lr * b.grad

                # clean grad
                w.grad.zero_()
                b.grad.zero_()

                # record loss value
                losses.append(loss.item())
    # draw
    plt.figure()
    plt.plot(range(len(losses)), losses, color = 'red',label='Train')
    # plt.plot(range(len(self.losses_test)), self.losses_test, color = 'blue',label='Test')
    plt.title("Convergence Graph of Loss Function")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.show()

    # 这里还没有加入测试代码，一般来说，测试代码有几种情况
    # 1. test once per serval batch
    # 2. test onece per epoch
    # 3. epoch VS step