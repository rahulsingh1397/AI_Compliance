import torch
import torch.nn as nn

class SimpleRiskModel(nn.Module):
    """A simple feed-forward neural network to simulate a risk classification model."""
    def __init__(self, input_size=10, hidden_size=16, output_size=1):
        super(SimpleRiskModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Defines the forward pass of the model."""
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out) # Output a value between 0 and 1 (e.g., risk probability)
        return out

def create_model_and_input(input_size=10):
    """Helper function to create an instance of the model and a dummy input tensor."""
    model = SimpleRiskModel(input_size=input_size)
    # Create a dummy input tensor for exporting the model
    dummy_input = torch.randn(1, input_size, requires_grad=False)
    return model, dummy_input
