import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# 1 color channel, 6 filters, 3by3 kernel, stride = 1
conv1 = nn.Conv2d(1,6,3,1) # ---> 6 filters ---> pooling ---> conv2

# 6 input Filters Conv1, 16 filters, 3by3, stride = 1
conv2 = nn.Conv2d(6,16,3,1)

for i, (X_train, y_train) in enumerate(train_data):
    break

x = X_train.view(1,1,28,28) # ---> 4D batch (batch of 1 image)

x = F.relu(conv1(x))
x = F.max_pool2d(x, 2, 2)
x = F.relu(conv2(x))
x = F.max_pool2d(x, 2, 2)

x.view(-1, 16 *5 * 5)

class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)

torch.manual_seed(42)
model = ConvolutionalNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

if __name__ == '__main__':
    start_time = time.time()

    # Variables (Trackers
    epochs = 5
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    # For loop epochs
    for i in range(epochs):

        trn_corr = 0
        tst_corr = 0

        # Train
        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1

            y_pred = model(X_train)  # Not flattening
            loss = criterion(y_pred, y_train)

            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 600 == 0:
                print(f'Epoch: {i} Batch: {b} Loss: {loss.item()}')

        train_losses.append(loss)
        train_correct.append(trn_corr)


        # Test

        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):

                y_val = model(X_test)

                predicted = torch.max(y_val.data,1)[1]
                tst_corr += (predicted == y_test).sum()

        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)

    current_time = time.time()
    total = current_time - start_time
    print(f'Training took {total/60} minutes.')


