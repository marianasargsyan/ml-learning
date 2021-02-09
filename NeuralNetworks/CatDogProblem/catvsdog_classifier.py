import torch
import argparse
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.metrics import accuracy_score

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def pars_args():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, help='dataset path.', required=True)
    return args.parse_args()


input_size = 200
params = pars_args()
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(input_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       T.Lambda(lambda x: torch.flatten(x))])
test_transforms = transforms.Compose([transforms.Resize(input_size),
                                      transforms.ToTensor(),
                                      T.Lambda(lambda x: torch.flatten(x))])


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size ** 2 * 3, 1024)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        return F.log_softmax(x)


train_data = datasets.ImageFolder(params.dataset + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(params.dataset + '/valid', transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

model = Network()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.BCELoss()

for i, (data, labels) in enumerate(train_loader):
    data.to('cuda')
    labels.to('cuda')
    out = model(data)
    loss = loss_func(out.reshape(-1).to(torch.float32), labels.to(torch.float32))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 100:
        print(loss, accuracy_score(labels.data.cpu().detach().numpy(), out.data.cpu().detach().numpy()))
