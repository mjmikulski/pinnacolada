import torch
from torch.nn import functional as F, Softplus


class TwoSkips(torch.nn.Module):
    """

    Inputs:
    - t
    - v_0

    Outputs:
    - x
    - x.
    - x..
    - alpha
    - beta


    """

    def __init__(self, bus_dim=16, output_dim=5):
        super().__init__()
        self.bus_dim = bus_dim

        # self.activation = torch.nn.ELU()
        # self.activation = torch.nn.LeakyReLU()
        self.activation = torch.tanh
        self.fc1 = torch.nn.Linear(bus_dim, 2 * bus_dim)
        self.fc1t = torch.nn.Linear(bus_dim, 2 * bus_dim)
        self.fc2 = torch.nn.Linear(4 * bus_dim, bus_dim)
        self.fc3 = torch.nn.Linear(bus_dim, 2 * bus_dim)
        self.fc3t = torch.nn.Linear(bus_dim, 2 * bus_dim)
        self.fc4 = torch.nn.Linear(4 * bus_dim, bus_dim)
        self.fc5a = torch.nn.Linear(bus_dim, 3)
        self.fc5b = torch.nn.Linear(bus_dim, 2)

    def forward(self, t, v_0):
        bs = t.shape[0]
        bd = self.bus_dim - 2

        t = t.reshape((bs, 1))
        v_0 = v_0.reshape((bs, 1))

        x_in = torch.concat([t, v_0, torch.zeros((bs, bd))], dim=1)

        a = self.activation(self.fc1(x_in))
        b = self.fc1t(x_in) * t
        c = torch.concat([a, b], dim=1)

        # x = x_in + self.activation(self.fc2(c))
        x = x_in + self.fc2(c)

        a = self.activation(self.fc3(x))
        b = self.fc3t(x) * t
        c = torch.concat([a, b], dim=1)

        # x = x + self.activation(self.fc4(c))
        x = x + self.fc4(c)

        a = self.fc5a(x)
        b = F.softplus(self.fc5b(x))
        c = torch.concat([a, b], dim=1)

        return c
