import linear_regression.linear_regression as linear_regression
from torchsummary import summary
from utils import split_dataset, plot_feature_vs_target, create_dataloader, plot_losses, create_dataloaders, plot_accuracies

def run(X, args, device, test_loader, train_loader, val_loader):
    model = linear_regression.LinearRegression(input_size=X.shape[1])
    model = model.to(device)
    summary(model, (X.shape[1],))
    train_losses, val_losses = linear_regression.train_model(model, train_loader=train_loader, val_loader=val_loader,
                                                             num_epochs=args.epochs, lr=args.lr)
    plot_losses(train_losses, val_losses, file_name=f"{args.model.lower()}-loss.png")
    test_loss = linear_regression.test_model(model, test_loader)
    print(f"Test Mean Squared Error: {test_loss:.4f}")
    # Evaluation Metric
    linear_regression.get_model_parameters(model=model)
    linear_regression.evaluation_metrics(model, test_loader, n_features=X.shape[1])
    return