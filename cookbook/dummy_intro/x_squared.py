import torch
from torch.optim import SGD

from cookbook.dummy_intro.model import Model

if __name__ == '__main__':
    x = torch.linspace(-1, 1, 100).view(-1, 1)
    x.requires_grad = True
    y_true = 0.5 * x ** 2

    model = Model()

    optimizer = SGD(model.parameters(), lr=0.001)

    for epoch in range(2001):
        optimizer.zero_grad()
        y = model(x)

        dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

        # We don't use y_true here but only the fact that derivative of `0.5 * x ^ 2` equals `x`
        loss = torch.sum((dy_dx - x) ** 2)

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch: {epoch:04d}\tloss: {loss.item():.4g}')
