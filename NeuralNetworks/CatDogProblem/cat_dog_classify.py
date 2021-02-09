import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_train = torch.utils.data.DataLoader('dataset/train',
                                          batch_size=20,
                                          shuffle=True)
data_valid = torch.utils.data.DataLoader('dataset/valid',
                                          batch_size=1,
                                          shuffle=True)

print(data_valid)
print(data_train)
# for i, (images, labels) in enumerate(data_valid):
#     print(type(images))
#
#
# print(data_valid)
# print(data_valid.size())
#