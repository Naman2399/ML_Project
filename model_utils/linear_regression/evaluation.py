import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *

def test(model, test_loader):
    criterion = nn.MSELoss()  # Mean squared error loss

    model.eval()
    test_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * len(inputs)
            num_samples += len(inputs)

    test_loss /= num_samples
    return test_loss