import torch
import torch.nn as nn
import numpy as np
import tqdm
import time
import argparse
import torchvision
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from sklearn.metrics import accuracy_score
from torch.utils.data.dataloader import DataLoader


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.input_size = 28
        self.flatten_size = 16 * 5 * 5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=self.flatten_size, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, X):
        y = self.pool1(torch.relu(self.conv1(X)))
        y = self.pool2(torch.relu(self.conv2(y)))
        y = y.view(-1, self.flatten_size)
        y = torch.tanh(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        return torch.softmax(self.output(y), dim=1)


def read(model, optim, lr_scheduler, initial_epoch, best_acc, metrics, model_path):
    if model_path == '':
        return model, optim, lr_scheduler, initial_epoch, best_acc, metrics
    _dict = torch.load(model_path)
    model.load_state_dict(_dict['model'])
    optim.load_state_dict(_dict['optimizer'])
    lr_scheduler.load_state_dict(_dict['scheduler'])
    initial_epoch = _dict['epoch']
    best_acc = _dict['accuracy']
    metrics = _dict['metrics']
    return model, optim, lr_scheduler, initial_epoch, best_acc, metrics


def write(model, optim, lr_scheduler, epoch, acc, metrics, model_path):
    checkpoints = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'scheduler': lr_scheduler.state_dict(),
        'accuracy': acc,
        'metrics': metrics
    }
    torch.save(checkpoints, model_path)


def test(model, test_loader, criterion, device):
    model.eval()
    outputs = []
    target_labels = []
    progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
    test_loss = 0.0

    for i, (data, target) in progress_bar:
        data = data.to(device)
        target = target.to(device)
        net_out = model(data)
        test_loss += criterion(net_out, target).item()
        outputs += list(net_out.cpu().detach().numpy().argmax(axis=1))
        target_labels += list(target.cpu().detach().numpy())
        if i % 100 == 0:
            progress_bar.set_description(
                "test loss: " + str((test_loss / (i + 1))) + " test acc:" + str(accuracy_score(outputs, target_labels)))

    outputs = np.reshape(outputs, -1)
    target_labels = np.reshape(target_labels, -1)

    return accuracy_score(outputs, target_labels), test_loss / len(test_loader)


def plot_results(metrics):
    fig, ax = plt.subplots(1, 4, figsize=(12, 6))
    ax = ax.ravel()
    s = ['Train loss', 'Train acc', 'Test loss', 'Test acc']
    for j in range(4):
        ax[j].plot(np.arange(len(metrics)), np.array(metrics)[:, j], marker='.', linewidth=1, markersize=6)
        ax[j].set_title(s[j])
    fig.tight_layout()
    fig.savefig('results.png', dpi=200)


def train(model, train_loader, test_loader, epochs, device):
    initial_epoch = 0
    best_acc = 0
    metrics = []
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 10)
    model, optim, lr_scheduler, initial_epoch, best_acc, metrics = read(model, optim, lr_scheduler, initial_epoch, best_acc, metrics, args.model)
    for e in range(initial_epoch, epochs):
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        acc = 0
        total_loss = 0.0
        model.train()
        for i, (data, target) in progress_bar:
            data = data.to(device)
            target = target.to(device)
            net_out = model(data)

            loss = criterion(net_out, target)
            loss.backward()
            total_loss += loss.item()
            optim.step()
            optim.zero_grad()
            acc += accuracy_score(target.cpu().detach().numpy(), net_out.cpu().detach().numpy().argmax(axis=1))
            if i % 100 == 0:
                progress_bar.set_description(
                    'epoch: ' + str(e + 1) + ' loss: ' + str(total_loss / (i + 1)) + ' acc: ' + str(acc / (i + 1)))
        lr_scheduler.step()
        test_acc, test_loss = test(model, test_loader, criterion, device)
        metrics.append([total_loss / len(train_loader), acc / len(train_loader), test_loss, test_acc])
        if test_acc > best_acc:
            best_acc = test_acc
            write(model, optim, lr_scheduler, e + 1, best_acc, metrics, model_path='best.pt')
        write(model, optim, lr_scheduler, e + 1, test_acc, metrics, model_path='last.pt')
        plot_results(metrics)


def pars_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = pars_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LeNet5()
    model.to(device)

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    mnist_trainset = MNIST(root='./data', train=True, download=True, transform=transforms)
    mnist_testset = MNIST(root='./data', train=False, download=True, transform=transforms)
    train_loader = DataLoader(mnist_trainset, args.batch_size, True)
    valid_loader = DataLoader(mnist_trainset, args.batch_size, True)

    start_time = time.time()
    train(model, train_loader, valid_loader, args.epochs, device)
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')