import torch
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def describe_dataset() :
    # Load the breast cancer dataset
    data = load_breast_cancer()

    # Extract features and target variable
    X = data.data  # Features
    y = data.target  # Target variable (0: malignant, 1: benign)
    y = y.reshape(-1, 1)

    # Find minimum and maximum values among all features
    min_value = np.min(X)
    max_value = np.max(X)

    # Print the minimum and maximum values among all features
    print("Minimum value among all features:", min_value)
    print("Maximum value among all features:", max_value)

    # Normalize the features to a scale of 0 to 1
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Find minimum and maximum values among all features
    min_value = np.min(X)
    max_value = np.max(X)

    # Print the minimum and maximum values among all features
    print("Minimum value among all features:", min_value)
    print("Maximum value among all features:", max_value)


    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    y = y.float()

    # Print the shape of the dataset
    print("Shape of features:", X.shape)
    print("Shape of target variable:", y.shape)

    return X, y

def load_dataset(args) :
    return describe_dataset()

