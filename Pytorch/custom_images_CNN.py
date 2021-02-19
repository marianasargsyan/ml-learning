import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import os
from PIL import Image
from IPython.display import display

import warnings

warnings.filterwarnings('ignore')

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root = 'Data/CATS_DOGS'

train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

torch.manual_seed(42)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10)

class_names = train_data.classes

for images, label in train_loader:
    break

im = make_grid(images, nrow=5)
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)
im_inv = inv_normalize(im)


class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54 * 54 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54 * 54 * 16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)


torch.manual_seed(101)
CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>8}')
    print(f'________\n{sum(params):>8}')


if __name__ == '__main__':
    print(class_names)
    plt.figure(figsize=(12, 4))
    plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
    plt.show()

    start_time = time.time()

    epochs = 3

    # LIMITS on num of batches
    max_trn_batch = 800
    max_tst_batch = 300

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for i in range(epochs):

        train_corr = 0
        test_corr = 0

        for b, (X_train, y_train) in enumerate(train_loader):

            if b == max_trn_batch:
                break

            b += 1

            y_pred = CNNmodel(X_train)
            loss = criterion(y_pred, y_train)
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            train_corr += batch_corr

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if b % 200 == 0:
                print(f'epoch: {i}  loss: {loss.item()} Acc {train_corr.item()*100/(10*b):7.3f}%')

        train_losses.append(loss)
        train_correct.append(train_corr)

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                # Limit the number of batches
                if b == max_tst_batch:
                    break

                # Apply the model
                y_val = CNNmodel(X_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                batch_corr += (predicted == y_test).sum()
                test_corr = test_corr + batch_corr
        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(test_corr)

    total_time = time.time() - start_time
    print(f'Total Time:  {total_time / 60} minutes')
