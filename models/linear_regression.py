import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *


class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=True)  # Linear layer with input_size features and 1 output, with bias

    def forward(self, x):
        # Forward pass: compute predicted y by passing x to the model
        return self.linear(x)
