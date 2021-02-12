import numpy as np
import matplotlib.pyplot as plt
import torch

X = torch.linspace(1, 50, 50).reshape(-1, 1)

e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2 * X + 1 + e

X = X.numpy()
y = y.numpy()
x_mean = np.mean(X)
y_mean = np.mean(y)


var_x = sum((X - x_mean)**2)/len(X)
var_y = sum((y - y_mean)**2)/len(y)

stdev_x = np.sqrt(var_x)
stdev_y = np.sqrt(var_y)

r = sum((X - x_mean)*(y - y_mean)/(stdev_x*stdev_y)) / len(X)

a = r * stdev_y / stdev_x

b = y_mean - (x_mean * a)

new_y = a * X + b

def cov(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    data = 0.0
    for i in range(len(x)):
        data += (x[i] - x_mean) * (y[i] - y_mean)

    return data / (len(data))


a1 = cov(X, y) / cov(X, X)
b1 = y_mean - (x_mean * a)
y_cov = a1 * X + b1

if __name__ == '__main__':
    # cov_xy = cov(X, y)
    # print(new_y)

    plt.scatter(X, y)
    plt.plot(new_y, 'r')
    plt.show()

    print(np.c_[new_y, y_cov])
