from models import linear_regression as linear_regression
from torchsummary import summary
from utils.data_utils import plot_losses
import torch
import torch.nn as nn
import torch.optim as optim
from model_utils.linear_regression.train import train
from model_utils.linear_regression.evaluation import test
from model_utils.linear_regression.utils import get_model_parameters
from model_utils.linear_regression.evaluation_meterics import evaluation_metrics

def run(model, X, args, device, test_loader, train_loader, val_loader):

    model = model.to(device)
    summary(model, (X.shape[1],))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Model Training and Validation
    train_losses, val_losses = train(model, train_loader=train_loader, val_loader=val_loader,
                                    num_epochs=args.epochs,
                                     criterion= criterion, optimizer= optimizer)

    # Plot losses
    plot_losses(train_losses, val_losses, file_name=f"{args.model.lower()}-loss.png")

    # Model Evaluation
    test_loss = test(model, test_loader)
    print(f"Test Mean Squared Error: {test_loss:.4f}")

    # Evaluation Metric
    get_model_parameters(model=model)
    evaluation_metrics(model, test_loader, n_features=X.shape[1])


    return