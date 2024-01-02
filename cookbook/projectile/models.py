import torch


class OneSkip(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        input_size = 4 + 1
        self.activation = torch.nn.ELU()
        self.fc1 = torch.nn.Linear(input_size, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, input_size)

    def forward(self, x, t):
        x_in = torch.concat([x, t], dim=1)
        x = self.activation(self.fc1(x_in)) * t
        x = self.activation(self.fc2(x))
        x = x_in + self.fc3(x)
        return x
