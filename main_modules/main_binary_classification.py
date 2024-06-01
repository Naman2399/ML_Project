from torchsummary import summary
from utils import split_dataset, plot_feature_vs_target, create_dataloader, plot_losses, create_dataloaders, plot_accuracies
import logisitc_regression.binary_classification as binary_class
from logisitc_regression import binary_classification


def run(X, args, device, test_loader, train_loader, val_loader):
    model = binary_class.BinaryClassifier(input_size=X.shape[1])
    model = model.to(device)
    summary(model, (X.shape[1],))
    train_losses, val_losses, train_accuracies, val_accuracies = binary_class.train_model(model,
                                                                                          train_loader=train_loader,
                                                                                          val_loader=val_loader,
                                                                                          num_epochs=args.epochs,
                                                                                          lr=args.lr)
    plot_losses(train_losses, val_losses, file_name=f"{args.model.lower()}-loss.png")
    plot_accuracies(train_accuracies, val_accuracies, file_name=f"{args.model.lower()}-loss.png")
    # Test and Evaluation Metric
    accuracy, _, _, _, _, average_loss = binary_classification.test_model(model, test_loader)
    print(f"Test Mean Squared Error: {average_loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}")
    return model