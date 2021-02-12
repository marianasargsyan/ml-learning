import numpy as np
import matplotlib.pyplot as plt
import torch

X = torch.linspace(1, 50, 50).reshape(-1, 1)
# print(X)


e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
# print(e)

y = 2 * X + 1 + e
# print(y.shape)

X = X.numpy()
y = y.numpy()
x_mean = np.mean(X)
y_mean = np.mean(y)

# plt.scatter(X, y)
# plt.show()


def cov(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    data = 0.0
    for i in range(len(x)):
        data += (x[i] - x_mean) * (y[i] - y_mean)

    return data / (len(data))


# print(cov1(X, y))

a = cov(X, y) / cov(X, X)

b = y_mean - (x_mean * a)

y_cov = a*X + b

cov_xy = cov(X, y)
print(y_cov)

plt.scatter(X, y)
plt.plot(y_cov, 'r')
plt.show()
