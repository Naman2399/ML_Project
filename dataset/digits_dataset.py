from sklearn.datasets import load_digits
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.preprocessing import OneHotEncoder


def describe_dataset():
    # Load the digits dataset
    data = load_digits()

    # Extract features and target variable
    X = data.data  # Features
    y = data.target  # Target variable

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

    # Convert numpy arrays to tensors
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # Print the unique labels
    unique_labels = torch.unique(y)
    print("Unique labels:", unique_labels)

    # Convert y to one-hot labels
    one_hot_encoder = OneHotEncoder(sparse_output=False)  # Explicitly set sparse=False
    y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1))
    y_one_hot = torch.tensor(y_one_hot, dtype=torch.float32)


    # Get details for 1 sample
    print("Details of Feature 1 : ", X[0])
    print("Labels of Feature 1 :", y_one_hot[0])

    # Print the shape of the dataset
    print("Shape of features:", X.shape)
    print("Shape of target variable:", y_one_hot.shape)

    return X, y_one_hot

def load_dataset() :
    return describe_dataset()

load_dataset()