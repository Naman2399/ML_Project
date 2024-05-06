import argparse

from utils import split_dataset, plot_feature_vs_target, create_dataloader, plot_losses
from torchsummary import summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe dataset details")

    # Dataset Description
    # 1. boston ---- housing price details
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g., 'housing')")
    parser.add_argument("--model", type=str, help="Name of the model to use (e.g. 'linear_reg', ")
    parser.add_argument("--batch", type=int, default= 256, help="Enter batch size ")
    parser.add_argument("--epochs", type=int, default=500, help="Enter number of epochs ")
    parser.add_argument("--lr", type=int, default=0.001, help="Learning Rate ")

    args = parser.parse_args()

    if args.dataset in ['housing']:
        if args.dataset.lower() == 'housing' :
            from dataset.housing_dataset import load_dataset
            X, y = load_dataset()
            # Call the function to plot each feature against the target and save them as images
            plot_feature_vs_target(X, y, output_dir="plots", file_prefix="features_vs_target")
    else :
        print("Dataset doesn't exits")
        print("Please provide a dataset name using the --dataset argument.")

    # Function to split the dataset
    X_train, y_train, X_test, y_test, X_val, y_val = split_dataset(X, y)

    if args.model in ['linear_reg'] :
        if args.model.lower() == 'linear_reg' :
            from linear_regression import linear_regression
            model = linear_regression.LinearRegression(input_size= X_train.shape[1])

    # Print model summary
    summary(model, (X_train.shape[1],))

    # Train, Test, Valid Loader
    train_loader = create_dataloader(X= X_train, y= y_train, batch_size= args.batch)
    test_loader = create_dataloader(X= X_test, y= y_test, batch_size= args.batch )
    validation_loader = create_dataloader(X= X_val, y= y_val, batch_size= args.batch)

    # Train Model
    if args.model in ['linear_reg'] :
        if args.model.lower() == 'linear_reg' :
            from linear_regression import linear_regression
            train_losses, val_losses = linear_regression.train_model(model= model, train_loader= train_loader, val_loader= validation_loader, num_epochs= args.epochs, lr= args.lr)

    # Plots
    plot_losses(train_losses= train_losses, val_losses= val_losses)

    # Test Model
    if args.model in ['linear_reg'] :
        if args.model.lower() == 'linear_reg' :
            from linear_regression import linear_regression
            test_loss = linear_regression.test_model(model, test_loader)
            print(f"Test Mean Squared Error: {test_loss:.4f}")

    # Evaluation Metrics
    if args.model in ['linear_reg'] :
        if args.model.lower() == 'linear_reg' :
            from linear_regression import linear_regression
            linear_regression.get_model_parameters(model= model)
            linear_regression.evaluation_metrics(model, test_loader, n_features= X_train.shape[1])











