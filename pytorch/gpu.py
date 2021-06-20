import torch
device = torch.device('cuda:0') # (cpu)
net = MLP().to(device) # in-place
optimizer = torch.optim.SGD(net.parameters)
criteon = torch.nn.CrossEntropyLoss().to(device)

for i in range(epoch):
    for features, labels in ds_train:
        features = features.to(device) # not in place
        labels = labels.to(device) # not in place