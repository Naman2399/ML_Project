import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader


def describe_dataset(batch_size = 256) :
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_data, train_labels = torch.tensor(trainset.data), torch.tensor(trainset.targets)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_data, test_labels = torch.tensor(testset.data), torch.tensor(testset.targets)

    print("Training data shape: ", train_data.shape)
    print("Training labels shape: ", train_labels.shape)
    print("Test data shape: ", test_data.shape)
    print("Test labels shape: ", test_labels.shape)

    # Combine the training and test data tensors
    combined_data = torch.cat((train_data, test_data), dim=0)
    combined_data = combined_data.permute(0, 3, 1, 2)
    combined_data = combined_data.float()
    combined_labels = torch.cat((train_labels, test_labels), dim=0)
    combined_labels = torch.nn.functional.one_hot(combined_labels.to(torch.int64), num_classes=10)


    print("Combined data shape: ", combined_data.shape)
    print("Combined labels shape: ", combined_labels.shape)

    return combined_data, combined_labels


def load_dataset() :
    return describe_dataset()
