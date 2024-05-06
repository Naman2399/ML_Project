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

# Define training function
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, update_frequency=20):
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []  # List to store training losses
    val_losses = []  # List to store validation losses

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': train_loss / len(train_loader)})
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        # Update progress bar description with validation loss
        if epoch % update_frequency == 0:
            progress_bar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})
        progress_bar.close()

    return train_losses, val_losses


def test_model(model, test_loader):
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


def get_model_parameters(model):
    """
    Get the parameters (weights and bias) of a PyTorch model.

    Args:
    - model: PyTorch model

    Returns:
    - weights: Tensor containing the weights of the model
    - bias: Tensor containing the bias of the model
    """
    parameters = model.state_dict()
    weights = parameters['linear.weight']
    bias = parameters['linear.bias']

    print("Weights:", weights)
    print("Bias:", bias)

    return weights, bias

def evaluation_metrics(model, test_loader, n_features):
    """
    Compute and print various evaluation metrics using the model and test loader.

    Args:
    - model: Trained regression model
    - test_loader: DataLoader for the test dataset
    - n_features: Number of features

    Returns:
    - None
    """
    # Call each evaluation metric function and print its result
    print("Mean Absolute Error (MAE):", mean_absolute_error(model, test_loader))
    print("Mean Squared Error (MSE):", mean_squared_error(model, test_loader))
    print("Root Mean Squared Error (RMSE):", root_mean_squared_error(model, test_loader))
    print("R-squared (R^2):", r_squared(model, test_loader))
    print("Adjusted R-squared (Adjusted R^2):", adjusted_r_squared(model, test_loader, n_features))