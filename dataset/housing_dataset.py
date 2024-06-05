from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import torch


def describe_dataset():
    # Load the California Housing dataset
    california_dataset = fetch_california_housing()
    # Extract features (X) and target variable (y) from the dataset
    X = california_dataset.data
    y = california_dataset.target.reshape(-1, 1)  # Reshape y to make it compatible with scaling

    # Remove 'Latitude' and 'Longitude' columns
    X_df = pd.DataFrame(X, columns=california_dataset.feature_names)
    X_df = X_df.drop(columns=['Latitude', 'Longitude'])

    # Normalize features to range [0, 1] using max scaling
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_df)
    y_scaled = scaler.fit_transform(y)

    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # Print basic details
    print("California Housing Dataset Details:")
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1] - 2)  # Subtracting 2 for 'Latitude' and 'Longitude'
    print("Feature names:", X_df.columns.tolist())

    # Display the first 10 rows of the dataset
    print("\nFirst 10 rows of the dataset:")
    print(X_df.head(10))

    # Display min-max values for each column
    print("\nMin-Max values for each column:")
    min_max_values = X_df.describe().loc[['min', 'max']]
    print(min_max_values)

    # Describe y using DataFrame
    y_df = pd.DataFrame(y_scaled, columns=['Target'])
    print("\nDescription of the target variable (y):")
    print(y_df.describe())

    # Display the first 10 rows of the dataset : y
    print("\nFirst 10 rows of the dataset: y labels")
    print(y_df.head(10))

    # Print shape for tensors
    print("X tensors : ", X_tensor.shape)
    print("Y tensors : ", y_tensor.shape)

    return X_tensor, y_tensor


def load_dataset(args) :
    return describe_dataset()