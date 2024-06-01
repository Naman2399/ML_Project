import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # One fully connected layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out