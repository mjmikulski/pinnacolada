import torch

from torch import nn
from torch.optim import SGD


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    x = torch.linspace(-1, 1, 100).view(-1, 1)
    x.requires_grad = True
    y_true = 0.5 * x ** 2

    model = Model()

    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(2001):
        optimizer.zero_grad()
        pred = model(x)

        dy_dx = torch.autograd.grad(pred, x, torch.ones_like(x), create_graph=True)[0]

        # We don't use y_true here but only the fact that derivative of `0.5 * x ^ 2` equals `x`
        loss = 0.01 * torch.sum((dy_dx - x) ** 2)

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch: {epoch:04d}\tloss: {loss.item():.4g}')
