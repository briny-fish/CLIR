import os
import numpy as np
import torch
import torch.nn as nn
import random
c = np.array([1,2,3,4,5,6])
d = torch.tensor([20.0])
e = torch.tensor([2.0])
a = torch.tensor([[1.0],[2.0],[3.0]])
b = torch.tensor([[2.0],[2.0],[1.0]])
print(d/e)
idx = range(1,5)
print(c[idx])
print(random.randint(range(10)))