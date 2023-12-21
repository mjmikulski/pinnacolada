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


USE_BOUNDARY_CONDITION = True

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
        physics_loss = torch.sum((dy_dx - x) ** 2)

        # Here we add boundary conditions
        if USE_BOUNDARY_CONDITION:
            boundary_loss = (pred[0] - y_true[0]) ** 2 + (pred[-1] - y_true[-1]) ** 2
        else:
            boundary_loss = torch.tensor(0.)

        alpha = 0.01  # In real life, we have to perform a few experiments to find the right value of alpha.
        loss = alpha * physics_loss + boundary_loss

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch: {epoch:04d}\tloss: {loss.item():.4g} '
                  f'\tphysics loss: {physics_loss.item():.4g} \tboundary loss: {boundary_loss.item():.4g}')

            # plot predictions
            import matplotlib.pyplot as plt

            x_numpy = x.detach().numpy()
            plt.plot(x_numpy, pred.detach().numpy(), label='pred')
            plt.plot(x_numpy, y_true.detach().numpy(), label='true')
            title_str = f'E: {epoch}  L: {loss.item():.3g}  phys: {physics_loss.item():.3g}  ' \
                        + (f'bound: {boundary_loss.item():.3g}' if USE_BOUNDARY_CONDITION else 'no boundary cond.')
            plt.title(title_str)
            plt.legend()
            filename = f'pinn_x_squared_{"boundary_" if USE_BOUNDARY_CONDITION else ""}{epoch:04d}.png'
            plt.savefig(filename)
            plt.close()
