import numpy as np
from PIL import Image

x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
d = np.sqrt(x * x + y * y)
sigma, mu = 1.0, 0.0
g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
print("2D Gaussian-like array:")
print(type(g))


img = Image.fromarray(g, 'RGB')
img.save('my.png')
img.show()

