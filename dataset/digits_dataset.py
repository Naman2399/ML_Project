from sklearn.datasets import load_digits
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

    # Print the shape of the dataset
    print("Shape of features:", X.shape)
    print("Shape of target variable:", y.shape)

    return X, y

def load_dataset() :
    return describe_dataset()
