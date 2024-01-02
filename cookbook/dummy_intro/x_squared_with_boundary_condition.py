import torch
from torch.optim import SGD

from cookbook.dummy_intro.model import Model

USE_BOUNDARY_CONDITION = True

if __name__ == '__main__':
    x = torch.linspace(-1, 1, 100).view(-1, 1)
    x.requires_grad = True
    y_true = 0.5 * x ** 2

    model = Model()

    optimizer = SGD(model.parameters(), lr=0.05)

    for epoch in range(2001):
        optimizer.zero_grad()
        y = model(x)

        dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

        # We don't use y_true here but only the fact that derivative of `0.5 * x ^ 2` equals `x`
        physics_loss = torch.sum((dy_dx - x) ** 2)

        # Here we add boundary conditions
        if USE_BOUNDARY_CONDITION:
            boundary_loss = (y[0] - y_true[0]) ** 2 + (y[-1] - y_true[-1]) ** 2
            # You can also try experimenting with other boundary conditions, e.g.:
            # value at x=0.
        else:
            boundary_loss = torch.tensor(0.)

        alpha = 0.02  # In real life, we have to perform a few experiments to find the right value of alpha.
        loss = alpha * physics_loss + boundary_loss

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch: {epoch:04d}\tloss: {loss.item():.4g} '
                  f'\tphysics loss: {physics_loss.item():.4g} \tboundary loss: {boundary_loss.item():.4g}')

            # plot predictions
            import matplotlib.pyplot as plt

            x_numpy = x.detach().numpy()
            plt.plot(x_numpy, y.detach().numpy(), label='pred')
            plt.plot(x_numpy, y_true.detach().numpy(), label='true')
            title_str = f'E: {epoch}  L: {loss.item():.3g}  phys: {physics_loss.item():.3g}  ' \
                        + (f'bound: {boundary_loss.item():.3g}' if USE_BOUNDARY_CONDITION else 'no boundary cond.')
            plt.title(title_str)
            plt.legend()
            filename = f'pinn_x_squared_{"boundary_" if USE_BOUNDARY_CONDITION else ""}{epoch:04d}.png'
            plt.savefig(filename)
            plt.close()
