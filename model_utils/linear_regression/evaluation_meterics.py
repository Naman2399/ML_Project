import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *

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