import torch
import torch.nn as nn


# Always use softplus as activation function for MLPs
class network_prim(nn.Module):
    def __init__(self, network_length, input_dimension, hidden_dimension, output_dimension):
        super().__init__()

        self.network_length = network_length
        self.linears = nn.ModuleList([nn.Linear(input_dimension, hidden_dimension)])
        self.linears.extend(
            [nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1, network_length - 1)]
        )
        self.linears.extend([nn.Linear(hidden_dimension, output_dimension, bias=False)])
        self.softplus = nn.Softplus()

    def initialization(self):
        for layer in self.linears:
            layer.weight.data.normal_()
            if layer.bias is not None:
                layer.bias.data.normal_()

    def forward(self, x):
        layer = self.linears[0]
        x = layer(x)
        for layer in self.linears[1 : self.network_length - 1]:
            x = self.softplus(x)
            x = layer(x)

        x = self.softplus(x)
        layer = self.linears[self.network_length - 1]
        x = layer(x)

        return x


class network_dual(nn.Module):
    def __init__(self, network_length, input_dimension, hidden_dimension, output_dimension, L):
        super().__init__()

        self.network_length = network_length
        self.linears = nn.ModuleList([nn.Linear(input_dimension, hidden_dimension)])
        self.linears.extend(
            [nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1, network_length - 1)]
        )
        self.linears.extend([nn.Linear(hidden_dimension, output_dimension)])
        self.softplus = nn.Softplus()
        self.L = L

    def initialization(self):
        for layer in self.linears:
            layer.weight.data.normal_()
            if layer.bias is not None:
                layer.bias.data.normal_()

    def modif_function(self, x):
        modif_fun = torch.min(torch.min(x + self.L / 2, self.L / 2 - x), -1)[0]
        modif_fun = modif_fun.unsqueeze(-1)
        return modif_fun

    def forward(self, x):
        modify = self.modif_function(x)
        layer = self.linears[0]
        x = layer(x)
        for layer in self.linears[1 : self.network_length - 1]:
            x = self.softplus(x)
            x = layer(x)

        x = self.softplus(x)
        layer = self.linears[self.network_length - 1]
        x = layer(x)
        x = x * modify

        return x


class network_dual_on_bdry(nn.Module):
    def __init__(self, network_length, input_dimension, hidden_dimension, output_dimension):
        super().__init__()

        self.network_length = network_length
        self.linears = nn.ModuleList([nn.Linear(input_dimension, hidden_dimension)])
        self.linears.extend(
            [nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1, network_length - 1)]
        )
        self.linears.extend([nn.Linear(hidden_dimension, output_dimension, bias=False)])
        self.softplus = nn.Softplus()

    def initialization(self):
        for layer in self.linears:
            layer.weight.data.normal_()
            if layer.bias is not None:
                layer.bias.data.normal_()

    def forward(self, x):
        layer = self.linears[0]
        x = layer(x)
        for layer in self.linears[1 : self.network_length - 1]:
            x = self.softplus(x)
            x = layer(x)

        x = self.softplus(x)
        layer = self.linears[self.network_length - 1]
        x = layer(x)

        return x
