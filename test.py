import os
import numpy as np
import torch
import torch.nn as nn
import random
import time
import datetime
import pandas as pd
import dataLoader
a = torch.FloatTensor([[[1,2,3,3],[1,2,3,4],[1,2,3,4]],[[2,2,3,3],[1,3,3,4],[1,2,3,4]]])
a.requires_grad = True

b = torch.FloatTensor([[1,2,2,2],[2,1,1,1]])
bb = torch.norm(b,dim = -1)
aa = torch.norm(a,dim = -1)
print(aa,bb)
out1 = torch.einsum("ijk,ik->ij",[a,b])
print(out1)
print(torch.einsum("ij,i->ij",[aa,bb]))
print(out1/torch.einsum("ij,i->ij",[aa,bb]))
c = a.view(2,-1)
out = c.mean()
print(out)

d = {1:1,2:2}
out.backward()
print(a.grad)
out1.backward()
print(a.grad)
