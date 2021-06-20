import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

class LinearRegression(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.params = torch.nn.ParameterDict({
            'w': torch.nn.Parameter(torch.zeros(in_features, out_features)),
            'b': torch.nn.Parameter(torch.zeros(1))
        })
    def forward(self, x):
        return x @ self.params['w'] + self.params['b']

if __name__ == '__main__':
    net = LinearRegression(10, 1)
    for a in net.parameters():
        print(a)
    # print(net.parameters())
if __name__ == "__main2__":
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
    net = LinearRegression(X.shape[1], 1)
    lr = torch.tensor(0.05)
    losses = []

    # train
    epoch = 100
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    for i in range(epoch):
        for features, labels in dl_train:
            # forward
            y_pred = net(features)
            loss = loss_func(y_pred, labels)
            
            # backward
            loss.backward()
            # update parameters
            optimizer.step()
            # clean grad
            optimizer.zero_grad()
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