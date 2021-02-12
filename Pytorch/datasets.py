import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv('Data/iris.csv')

print(df.head())

features = df.drop('target', axis=1).values
label = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2,random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train).reshape(-1,1)
y_test = torch.LongTensor(y_test).reshape(-1,1)

data = df.drop('target', axis=1).values
labels = df['target'].values

iris = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
# print(type(iris))
# print(len(iris))

# for i in iris:
#     print(i)

iris_loader = DataLoader(iris, batch_size=50, shuffle=True)

for i_batch, sample_batch in enumerate(iris_loader):
    print(i_batch, sample_batch)