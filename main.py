import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from logisitc_regression import binary_classification
from model_utils.device import get_available_device
from utils import split_dataset, plot_feature_vs_target, create_dataloader, plot_losses, create_dataloaders, plot_accuracies
import dataset.housing_dataset as housing
import dataset.breast_cancer_dataset as breast_cancer
import dataset.digits_dataset as digits
import dataset.cifar10 as cifar10
import linear_regression.linear_regression as linear_regression
import logisitc_regression.binary_classification as binary_class
import logisitc_regression.multiclass_classification as multi_class
import models.lenet_5 as lenet_5
import model_utils.train as training
import model_utils.evaluation as evaluation

import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")

def load_dataset(name)  :
    datasets = {
        'housing': housing.load_dataset,
        'breast_cancer': breast_cancer.load_dataset,
        'digits': digits.load_dataset,
        'cifar10': cifar10.load_dataset
    }
    return datasets[name]()

def main():
    parser = argparse.ArgumentParser(description="Describe dataset details")
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g., 'housing', 'breast_cancer', 'digits', 'cifar10')")
    parser.add_argument("--model", type=str, help="Name of the model to use (e.g., 'linear_reg', 'binary_class', 'multi_class', 'lenet')")
    parser.add_argument("--batch", type=int, default=256, help="Enter batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Enter number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    args = parser.parse_args()


    '''
    Device details
    '''
    # Example usage
    device = get_available_device()
    print(f"Using device: {device}")

    '''
    Load Dataset 
    '''
    if args.dataset.lower() in ['housing', 'breast_cancer', 'digits', 'cifar10']:
        X, y = load_dataset(args.dataset.lower())
    else:
        print("Dataset doesn't exist")
        print("Please provide a dataset name using the --dataset argument.")
        return

    X = X.to(device)
    y = y.to(device)
    train_loader, test_loader, val_loader = create_dataloaders(X, y, batch_size=args.batch, test_frac=0.1, val_frac=0.1)


    '''
    Extras 
    '''
    if args.dataset.lower() == 'housing':
        plot_feature_vs_target(X, y, output_dir="plots", file_prefix="features_vs_target")


    '''
    Models 
    '''
    if args.model.lower() in ['linear_reg']:
        model = linear_regression.LinearRegression(input_size=X.shape[1])
        model = model.to(device)
        summary(model, (X.shape[1], ))
        train_losses, val_losses = linear_regression.train_model(model, train_loader= train_loader, val_loader= val_loader, num_epochs= args.epochs, lr= args.lr)
        plot_losses(train_losses, val_losses, file_name=f"{args.model.lower()}-loss.png")
        test_loss = linear_regression.test_model(model, test_loader)
        print(f"Test Mean Squared Error: {test_loss:.4f}")
        # Evaluation Metric
        linear_regression.get_model_parameters(model=model)
        linear_regression.evaluation_metrics(model, test_loader, n_features=X.shape[1])

    if args.model.lower() in ['binary_class'] :
        model = binary_class.BinaryClassifier(input_size=X.shape[1])
        model = model.to(device)
        summary(model, (X.shape[1],))
        train_losses, val_losses, train_accuracies, val_accuracies = binary_class.train_model(model, train_loader= train_loader, val_loader= val_loader, num_epochs= args.epochs, lr= args.lr)
        plot_losses(train_losses, val_losses, file_name=f"{args.model.lower()}-loss.png")
        plot_accuracies(train_accuracies,val_accuracies,  file_name=f"{args.model.lower()}-loss.png")
        # Test and Evaluation Metric
        accuracy, _, _, _, _, average_loss = binary_classification.test_model(model, test_loader)
        print(f"Test Mean Squared Error: {average_loss:.4f}")
        print(f"Test Accuracy : {accuracy:.4f}")

    if args.model.lower() in ['multi_class'] :
        model = multi_class.SimpleNN(input_size=X.shape[1], num_classes= y.shape[1])
        model = model.to(device)
        summary(model, (X.shape[1],))
        train_losses, val_losses, train_accuracies, val_accuracies = multi_class.train_model(model,
                                        train_loader=train_loader, val_loader=val_loader, num_epochs=args.epochs, lr=args.lr)
        plot_losses(train_losses, val_losses, file_name=f"{args.model.lower()}-loss.png")
        plot_accuracies(train_accuracies, val_accuracies,  file_name=f"{args.model.lower()}-acc.png")
        # Test and Evaluation Metric
        accuracy, _, _, _, _, average_loss = multi_class.test_model(model, test_loader, num_classes= y.shape[1])
        print(f"Test Mean Squared Error: {average_loss:.4f}")
        print(f"Test Accuracy : {accuracy:.4f}")

    if args.model.lower() in ['lenet'] :
        model = lenet_5.LeNet5()
        model.to(device)
        summary(model, (X.shape[1], X.shape[2], X.shape[3], ))

        input_tensor_example = torch.rand(((X.shape[1], X.shape[2], X.shape[3])))
        print("Input Tensor Example : ", input_tensor_example.shape)
        writer.add_graph(model, input_tensor_example)
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Model Training and Validation
        import model_utils.train as training
        train_losses, train_accuracies, val_losses, val_accuracies = training.train(model,
                    train_loader=train_loader, validation_loader=val_loader, criterion=criterion,
                    optimizer=optimizer, epochs=args.epochs, writer= writer)

        plot_losses(train_losses=train_losses, val_losses=val_losses, file_name=f"{args.model.lower()}-loss.png")
        plot_accuracies(train_accuracies, val_accuracies, file_name=f"{args.model.lower()}-acc.png")

        # Test and evaluation and confusion matrix
        import model_utils.evaluation as evaluation
        criterion = nn.CrossEntropyLoss()
        test_accuracy, test_loss, confusion = evaluation.evaluate(model, test_loader, criterion=criterion)
        print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")
        print("Confusion Matrix:")
        print(confusion)

    writer.close()
    sys.exit()


if __name__ == "__main__":
    main()