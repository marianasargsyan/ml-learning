import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from PIL import Image
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

# Check the full file path

# with Image.open('Data/CATS_DOGS/test/CAT/9391.jpg') as im:
#     display(im)

path = 'Data/CATS_DOGS/'
img_names = []

for folder, subfolder, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder+'/'+img)

# print(len(img_names))

img_sizes = []
rejected = []

for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)

    except:
        rejected.append(item)

# print(len(rejected))

df = pd.DataFrame(img_sizes)
# print(df[0].describe())

dog = Image.open('Data/CATS_DOGS/train/DOG/14.jpg')

display(dog)
# print(dog.getpixel((0, 0)))

transform = transforms.Compose([
    transforms.Resize(250),
    transforms.CenterCrop(250),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])
#
# im = transform(dog)
# print(type(im))
# print(im.shape)
#
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

im = transform(dog)
print(type(im))
print(im.shape)

plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.show()