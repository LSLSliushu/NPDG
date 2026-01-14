import torch
import torch.nn as nn


# primal nn
class network_prim(nn.Module):
    def __init__(self, network_length, input_dimension, hidden_dimension, output_dimension):
        super(network_prim, self).__init__()
        self.network_length = network_length
        self.linears = nn.ModuleList([nn.Linear(input_dimension, hidden_dimension)])
        self.linears.extend([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1, network_length-1)])
        self.linears.extend([nn.Linear(hidden_dimension, output_dimension, bias=False)])
        self.tanh = nn.Tanh()

    def initialization(self):
        for l in self.linears:
            l.weight.data.normal_()
            if l.bias is not None:
                l.bias.data.normal_()

    def forward(self, x):
        l = self.linears[0]
        x = l(x)
        for l in self.linears[1: self.network_length-1]:
            x = self.tanh(x)
            x = l(x)
        x = self.tanh(x)
        l = self.linears[self.network_length-1]
        x = l(x)
        return x


# dual nn
# length = 8, input_dim = dim, hidden_dim = 50, output_dim = 1
class network_dual(nn.Module):
    def __init__(self, network_length, input_dimension, hidden_dimension, output_dimension, R):
        super(network_dual, self).__init__()

        self.network_length = network_length

        self.linears = nn.ModuleList([nn.Linear(input_dimension, hidden_dimension)])
        self.linears.extend([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1, network_length-1)])
        self.linears.extend([nn.Linear(hidden_dimension, output_dimension)])
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.R = R


    def initialization(self):
        for l in self.linears:
            l.weight.data.normal_()
            if l.bias is not None:
                l.bias.data.normal_()

    def modif_function(self, x):
        modif_fun = self.R**2 - torch.sum(x * x, -1).unsqueeze(-1)
        return modif_fun

    def forward(self, x):
        modify = self.modif_function(x)
        l = self.linears[0]
        x = l(x)
        for l in self.linears[1: self.network_length-1]:
            x = self.tanh(x)
            x = l(x)
        x = self.tanh(x)
        l = self.linears[self.network_length-1]
        x = l(x)
        x = x * modify
        return x


class network_dual_on_bdry(nn.Module):
    def __init__(self, network_length, input_dimension, hidden_dimension, output_dimension):
        super(network_dual_on_bdry, self).__init__()
        self.network_length = network_length
        self.linears = nn.ModuleList([nn.Linear(input_dimension, hidden_dimension)])
        self.linears.extend([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1, network_length-1)])
        self.linears.extend([nn.Linear(hidden_dimension, output_dimension, bias=False)])
        self.tanh = nn.Tanh()

    def initialization(self):
        for l in self.linears:
            l.weight.data.normal_()
            if l.bias is not None:
                l.bias.data.normal_()

    def forward(self, x):
        l = self.linears[0]
        x = l(x)
        for l in self.linears[1: self.network_length-1]:
            x = self.tanh(x)
            x = l(x)
        x = self.tanh(x)
        l = self.linears[self.network_length-1]
        x = l(x)
        return x


