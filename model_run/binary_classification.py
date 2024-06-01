from torchsummary import summary
from utils import plot_losses, plot_accuracies
import models.binary_classification as binary_class
from models import binary_classification
import torch.nn as nn
import torch.optim as optim
from model_utils.binary_classification.train import train
from model_utils.binary_classification.evaluation import test

def run(model, X, args, device, test_loader, train_loader, val_loader):

    model = model.to(device)
    summary(model, (X.shape[1],))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Model Training and Validation
    train_losses, val_losses, train_accuracies, val_accuracies = train(model,
                                train_loader=train_loader, val_loader=val_loader,
                                num_epochs=args.epochs, criterion= criterion, optimizer= optimizer)


    plot_losses(train_losses, val_losses, file_name=f"{args.model.lower()}-loss.png")
    plot_accuracies(train_accuracies, val_accuracies, file_name=f"{args.model.lower()}-loss.png")

    # Test and Evaluation Metric
    accuracy, _, _, _, _, average_loss = test(model, test_loader)
    print(f"Test Mean Squared Error: {average_loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}")
    return model