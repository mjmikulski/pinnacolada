"""
Refs
---
https://stackoverflow.com/q/73698041/8788960
"""

import torch

a = torch.tensor([5.], requires_grad=True)
b = a * a
c = torch.sin(b)

b.retain_grad()
c.retain_grad()

c.backward(create_graph=True)

assert a.grad.requires_grad is True
assert a.grad.is_leaf is False

assert b.grad.requires_grad is True
assert b.grad.is_leaf is False

del a, b, c
# dupa

a = torch.tensor([5.], requires_grad=True)
b = torch.tensor([5.], requires_grad=True)
c = torch.tensor([5.], requires_grad=False)

d = torch.sin(a * b * c)

assert d.requires_grad == any((x.requires_grad for x in (a, b, c)))

