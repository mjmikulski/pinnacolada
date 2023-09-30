"""
Refs
---
https://stackoverflow.com/q/73698041/8788960
"""

import torch

a = torch.tensor([1.], requires_grad=True)
y = torch.zeros((10))  # y is a non-leaf tensor
print(f'{y.is_leaf=}')

y[0] = a
print(f'{y.is_leaf=}')
y.retain_grad()
print(f'{y.is_leaf=}')
y[1] = y[0] * 2
print(f'{y.is_leaf=}')

print('ok')