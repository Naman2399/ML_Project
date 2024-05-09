import argparse

from utils import split_dataset, plot_feature_vs_target, create_dataloader, plot_losses, create_dataloaders, \
    plot_accuracies
from torchsummary import summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe dataset details")

    # Dataset Description
    # 1. boston ---- housing price details
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g., 'housing', 'breast_cancer', 'digits')")
    parser.add_argument("--model", type=str, help="Name of the model to use (e.g. 'linear_reg', 'binary_class', 'multi_class) ")
    parser.add_argument("--batch", type=int, default= 256, help="Enter batch size ")
    parser.add_argument("--epochs", type=int, default=100, help="Enter number of epochs ")
    parser.add_argument("--lr", type=int, default=0.001, help="Learning Rate ")

    args = parser.parse_args()

    if args.dataset in ['housing', 'breast_cancer', 'digits']:
        if args.dataset.lower() == 'housing' :
            from dataset.housing_dataset import load_dataset
            X, y = load_dataset()
            # Call the function to plot each feature against the target and save them as images
            plot_feature_vs_target(X, y, output_dir="plots", file_prefix="features_vs_target")

        if args.dataset.lower() == 'breast_cancer' :
            from dataset.breast_cancer_dataset import load_dataset
            X, y = load_dataset()

        if args.dataset.lower() == 'digits' :
            from dataset.digits_dataset import load_dataset
            X, y = load_dataset()

    else :
        print("Dataset doesn't exits")
        print("Please provide a dataset name using the --dataset argument.")

    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(X, y, batch_size=32, test_frac=0.1,
                                                                           val_frac=0.1)

    if args.model in ['linear_reg' ,'binary_class', 'multi_class']:
        if args.model.lower() == 'linear_reg' :
            from linear_regression import linear_regression
            model = linear_regression.LinearRegression(input_size= X.shape[1])

        if args.model.lower() == 'binary_class' :
            from logisitc_regression import binary_classification
            model = binary_classification.BinaryClassifier(input_size= X.shape[1])


        if args.model.lower() == 'multi_class' :
            from logisitc_regression import multiclass_classification


    # Print model summary
    summary(model, (X.shape[1],))

    # Train Model
    if args.model in ['linear_reg', 'binary_class', 'multi_class'] :
        if args.model.lower() == 'linear_reg' :
            from linear_regression import linear_regression
            train_losses, val_losses = linear_regression.train_model(model= model, train_loader= train_dataloader, val_loader= val_dataloader, num_epochs= args.epochs, lr= args.lr)
            # Plots Losses
            plot_losses(train_losses=train_losses, val_losses=val_losses, file_name= f"loss_{args.model.lower()}.png" )

        if args.model.lower() == 'binary_class' :
            from logisitc_regression import binary_classification
            train_losses, val_losses, train_accuracies, val_accuracies = binary_classification.train_model(model= model, train_loader= train_dataloader, val_loader= val_dataloader, num_epochs= args.epochs, lr= args.lr)
            # Plots Losses and Accuracies
            plot_losses(train_losses=train_losses, val_losses=val_losses, file_name= f"loss_{args.model.lower()}.png" )
            plot_accuracies(train_accuracies, val_accuracies, file_name= f"acc_{args.model.lower()}.png")


    # Test Model & Evaluation Meterics
    if args.model in ['linear_reg', 'binary_class', 'multi_class'] :
        if args.model.lower() == 'linear_reg' :
            # Test Model
            from linear_regression import linear_regression
            test_loss = linear_regression.test_model(model, test_dataloader)
            print(f"Test Mean Squared Error: {test_loss:.4f}")

            # Evaluation Metric
            from linear_regression import linear_regression
            linear_regression.get_model_parameters(model=model)
            linear_regression.evaluation_metrics(model, test_dataloader, n_features=X.shape[1])

        if args.model.lower() == 'binary_class' :
            # Test and Evaluation Metric
            from logisitc_regression import binary_classification
            accuracy, precision, recall, f1, roc_auc, average_loss = binary_classification.test_model(model, test_dataloader)
            print(f"Test Mean Squared Error: {average_loss:.4f}")
            print(f"Test Accuracy : {accuracy:.4f}")

