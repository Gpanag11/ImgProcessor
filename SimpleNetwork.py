import torch
import torch.nn as nn


class SimpleNetwork(nn.Module):

    def __init__(self, input_neurons: int, hidden_neurons: int, output_neurons: int,
                 activation_function: nn.Module = nn.ReLU()):
        super(SimpleNetwork, self).__init__()
        #fully-connected layers
        self.input_layer = nn.Linear(input_neurons, hidden_neurons)
        self.hidden_layer_1 = nn.Linear(hidden_neurons, hidden_neurons)
        self.hidden_layer_2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, output_neurons)

        #activation function
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the network
        x = self.activation_function(self.input_layer(x))
        x = self.activation_function(self.hidden_layer_1(x))
        x = self.activation_function(self.hidden_layer_2(x))
        x = self.output_layer(x)  # No activation function after the output layer
        return x


if __name__ == "__main__":
    torch.random.manual_seed(0)
    simple_network = SimpleNetwork(10, 20, 5)
    input_tensor = torch.randn(1, 10)
    output = simple_network(input_tensor)
    print(output)
