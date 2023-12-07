import torch
import torch.nn as nn

m = nn.Sigmoid()
loss = nn.BCELoss()
y = torch.randn(size=(5, 1), requires_grad=True)
print(y)
y = m(y)
print(y)
target = torch.ones(size=(5, 1))
print(target)
output = loss(y, target)
print(output)
