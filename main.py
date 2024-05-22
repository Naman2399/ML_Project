import argparse

from utils import split_dataset, plot_feature_vs_target, create_dataloader, plot_losses, create_dataloaders, \
    plot_accuracies
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim

import dataset.housing_dataset
import dataset.breast_cancer_dataset
import dataset.digits_dataset
import dataset.cifar10
import linear_regression.linear_regression as linear_regression
import logisitc_regression.binary_classification as binary_class
import logisitc_regression.multiclass_classification as multi_class
import models.lenet_5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe dataset details")

    # Dataset Description
    # 1. boston ---- housing price details
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g., 'housing', 'breast_cancer', 'digits', 'cifar10)")
    parser.add_argument("--model", type=str, help="Name of the model to use (e.g. 'linear_reg', 'binary_class', 'multi_class', 'lenet') ")
    parser.add_argument("--batch", type=int, default= 256, help="Enter batch size ")
    parser.add_argument("--epochs", type=int, default=100, help="Enter number of epochs ")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate ")

    args = parser.parse_args()

    if args.dataset.lower() in ['housing', 'breast_cancer', 'digits', 'cifar10']:
        dataset_loaders = {
            'housing': 'dataset.housing_dataset.load_dataset',
            'breast_cancer': 'dataset.breast_cancer_dataset.load_dataset',
            'digits': 'dataset.digits_dataset.load_dataset',
            'cifar10' : 'dataset.cifar10.load_dataset'
        }
        loader_function = dataset_loaders[args.dataset.lower()]
        X, y = eval(loader_function)()

    else :
        print("Dataset doesn't exits")
        print("Please provide a dataset name using the --dataset argument.")

    # Extra plots for linear regression
    if args.dataset.lower() == 'housing':
        plot_feature_vs_target(X, y, output_dir="plots", file_prefix="features_vs_target")

    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(X, y, batch_size=args.batch, test_frac=0.1,
                                                                           val_frac=0.1)

    if args.model.lower() in ['linear_reg', 'binary_class', 'multi_class', 'lenet']:
        model_constructors = {
            'linear_reg': 'linear_regression.LinearRegression(input_size=X.shape[1])',
            'binary_class': 'binary_class.BinaryClassifier(input_size=X.shape[1])',
            'multi_class': 'multi_class.SimpleNN(input_size=X.shape[1], num_classes=y.shape[1])',
            'lenet' : 'models.lenet_5.LeNet5()'
        }

        model_constructor = model_constructors[args.model.lower()]
        model = eval(model_constructor)


    # Print model summary
    if args.model.lower() in ['linear_reg', 'binary_class', 'multi_class']:
        summary(model, (X.shape[1],))
    if  args.model.lower() in ['lenet']:
        summary(model, ( X.shape[1], X.shape[2], X.shape[3], ))

    # Train Model
    if args.model.lower() in ['linear_reg', 'binary_class', 'multi_class']:
        from linear_regression import linear_regression
        from logisitc_regression import binary_classification, multiclass_classification

        model_constructors = {
            'linear_reg': (linear_regression.train_model, 'loss_linear_reg.png'),
            'binary_class': (binary_classification.train_model, 'loss_binary_class.png'),
            'multi_class': (multiclass_classification.train_model, 'loss_multi_class.png')
        }

        train_fn, loss_plot_name = model_constructors[args.model.lower()]
        # Classification problems
        if args.model.lower() == 'binary_class' or args.model.lower() == 'multi_class':
            train_losses, val_losses, train_accuracies, val_accuracies = train_fn(modely=model,
                                                                                  train_loader=train_dataloader,
                                                                                  val_loader=val_dataloader,
                                                                                  num_epochs=args.epochs, lr=args.lr)
            plot_accuracies(train_accuracies, val_accuracies, file_name=f"acc_{args.model.lower()}.png")
        # Regression Problem
        else:
            train_losses, val_losses = train_fn(model=model, train_loader=train_dataloader, val_loader=val_dataloader,
                                                num_epochs=args.epochs, lr=args.lr)

        plot_losses(train_losses=train_losses, val_losses=val_losses, file_name=loss_plot_name)

    if args.model.lower() in ['lenet'] :
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        import model_utils.train_loop as training
        train_losses, train_accuracies, val_losses, val_accuracies = training.train(model, train_loader= train_dataloader,
                        validation_loader= val_dataloader, criterion= criterion,
                       optimizer= optimizer, epochs= args.epochs)
        plot_accuracies(train_accuracies, val_accuracies, file_name=f"acc_{args.model.lower()}.png")
        plot_losses(train_losses=train_losses, val_losses=val_losses, file_name=f"loss_{args.model.lower()}.png")


    # Test Model
    if args.model.lower() in ['linear_reg', 'binary_class', 'multi_class', 'lenet']:
        if args.model.lower() == 'linear_reg':
            from linear_regression import linear_regression
            test_loss = linear_regression.test_model(model, test_dataloader)
            print(f"Test Mean Squared Error: {test_loss:.4f}")
            # Evaluation Metric
            linear_regression.get_model_parameters(model=model)
            linear_regression.evaluation_metrics(model, test_dataloader, n_features=X.shape[1])

        elif args.model.lower() == 'binary_class':
            # Test and Evaluation Metric
            from logisitc_regression import binary_classification

            accuracy, _, _, _, _, average_loss = binary_classification.test_model(model, test_dataloader)
            print(f"Test Mean Squared Error: {average_loss:.4f}")
            print(f"Test Accuracy : {accuracy:.4f}")

        elif args.model.lower() == 'multi_class':
            # Test and Evaluation Metric
            from logisitc_regression import multiclass_classification

            accuracy, _, _, _, _, average_loss = multiclass_classification.test_model(model, test_dataloader,
                                                                                      y.shape[1])
            print(f"Test Mean Squared Error: {average_loss:.4f}")
            print(f"Test Accuracy : {accuracy:.4f}")

        elif args.model.lower() == 'lenet' :
            # Test and evaluation
            import model_utils.evaluation as evaluation
            criterion = nn.CrossEntropyLoss()
            test_accuracy, test_loss, confusion = evaluation.evaluate(model, test_dataloader, criterion=criterion)
            print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")
            print("Confusion Matrix:")
            print(confusion)

