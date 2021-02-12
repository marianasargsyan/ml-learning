import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class Model(nn.Module):

    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        # how many layers?
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


torch.manual_seed(32)
model = Model()

df = pd.read_csv('Data/iris.csv')
# print(df.tail())

features = df.drop('target', axis=1).values
label = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2,random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):

    # Forward and get a prediction
    y_pred = model.forward(X_train)

    # calculate loss/error
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if i % 10 == 0:
        print(f'Epoch {i} and loss is: {loss}')

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# plt.plot(range(epochs), losses)
# plt.ylabel('LOSS')
# plt.xlabel('EPOCH')
# plt.show()

with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval,y_test)

print(loss)

correct = 0

with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        print(f'Data {i+1}.) {str(y_val)}')

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(correct)

torch.save(model.state_dict(), 'my_iris_model.pt')

new_model = Model()

new_model.load_state_dict(torch.load('my_iris_model.pt'))
print(new_model.eval())
