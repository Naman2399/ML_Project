from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import os


def create_dataloaders(X, y, test_frac=0.1, val_frac=0.1, batch_size=32):
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=42)

    # Split train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_frac, random_state=42)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.clone().detach(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.clone().detach(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.clone().detach(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.clone().detach(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.clone().detach(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.clone().detach(), dtype=torch.float32)

    # Create TensorDataset instances
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoader instances
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader, val_dataloader


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits the dataset into train, test, and validation sets.

    Parameters:
        X (torch.Tensor): Features tensor.
        y (torch.Tensor): Target tensor.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, X_val, y_val)
    """
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Further split the train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state)

    # Print the shapes of the resulting sets
    print("Train set shapes:", X_train.shape, y_train.shape)
    print("Test set shapes:", X_test.shape, y_test.shape)
    print("Validation set shapes:", X_val.shape, y_val.shape)

    return X_train, y_train, X_test, y_test, X_val, y_val




def plot_feature_vs_target(X, y, output_dir="plots", file_prefix="feature"):
    """
    Plot each feature against the target variable in scatter plots and save them as images.

    Parameters:
        X (torch.Tensor): Features tensor.
        y (torch.Tensor): Target tensor.
        output_dir (str): Directory to save the plots.
        file_prefix (str): Prefix for the file names of the saved plots.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_features = X.shape[1]
    for i in range(num_features):
        feature_values = X[:, i].numpy()
        plt.figure(figsize=(8, 6))
        plt.scatter(feature_values, y.numpy(), color='blue', alpha=0.5)
        plt.xlabel("Feature " + str(i + 1))
        plt.ylabel("Target")
        plt.title("Feature " + str(i + 1) + " vs. Target")
        # Save the plot as an image with the specified file name
        file_name = f"{file_prefix}_{i + 1}_vs_target.png"
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()

def create_dataloader(X, y, batch_size =  256 ) :

    # Create DataLoader for training set
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def plot_losses(train_losses, val_losses, output_dir="plots", file_name="loss_plot.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot as an image
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


def plot_accuracies(train_accuracies, val_accuracies, output_dir="plots", file_name="accuracy_plot.png"):
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot as an image
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()

def mean_absolute_error(model, test_loader):
    """
    Compute the mean absolute error (MAE) using the model and test loader.

    Args:
    - model: Trained regression model
    - test_loader: DataLoader for the test dataset

    Returns:
    - MAE: Mean Absolute Error
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            all_predictions.append(outputs.numpy())
            all_labels.append(labels.numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    return np.mean(np.abs(all_labels - all_predictions))


def mean_squared_error(model, test_loader):
    """
    Compute the mean squared error (MSE) using the model and test loader.

    Args:
    - model: Trained regression model
    - test_loader: DataLoader for the test dataset

    Returns:
    - MSE: Mean Squared Error
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            all_predictions.append(outputs.numpy())
            all_labels.append(labels.numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    return np.mean((all_labels - all_predictions) ** 2)


def root_mean_squared_error(model, test_loader):
    """
    Compute the root mean squared error (RMSE) using the model and test loader.

    Args:
    - model: Trained regression model
    - test_loader: DataLoader for the test dataset

    Returns:
    - RMSE: Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(model, test_loader))


def r_squared(model, test_loader):
    """
    Compute the coefficient of determination (R^2) using the model and test loader.

    Args:
    - model: Trained regression model
    - test_loader: DataLoader for the test dataset

    Returns:
    - R^2: Coefficient of Determination
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            all_predictions.append(outputs.numpy())
            all_labels.append(labels.numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    return r2_score(all_labels, all_predictions)


def adjusted_r_squared(model, test_loader, n_features):
    """
    Compute the adjusted coefficient of determination (adjusted R^2) using the model and test loader.

    Args:
    - model: Trained regression model
    - test_loader: DataLoader for the test dataset
    - n_features: Number of features

    Returns:
    - Adjusted R^2: Adjusted Coefficient of Determination
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            all_predictions.append(outputs.numpy())
            all_labels.append(labels.numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    n_samples = len(all_labels)
    R2 = r_squared(model, test_loader)
    adj_R2 = 1 - ((1 - R2) * (n_samples - 1) / (n_samples - n_features - 1))
    return adj_R2

