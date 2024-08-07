import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import model_utils.multi_class_image_classification.train as training
from utils.checkpoints import create_checkpoint_filename, load_checkpoint
from utils.data_utils import plot_losses, plot_accuracies


def run(X, args, device, model, test_loader, train_loader, val_loader, writer):

    model = model.to('cpu')
    summary(model, (X.shape[1], X.shape[2], X.shape[3]), device='cpu')
    model = model.to(device)
    # Tensorboard - adding model
    input_tensor_example = torch.rand(X.shape)
    input_tensor_example = input_tensor_example.to(device)
    print("Input Tensor Example : ", input_tensor_example.shape)
    writer.add_graph(model, input_tensor_example)



    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    current_epoch = 0
    least_val_loss = float('inf')

    # Checkpoint filename
    if args.ckpt_filename is not None :
        print("Loading contents from checkpoints")
        # Load model, optimizer, epoch from checkpoints
        checkpoint_filename = args.ckpt_filename
        model, optimizer, current_epoch, val_loss = load_checkpoint(model, optimizer, checkpoint_path=checkpoint_filename)
        current_epoch += 1
        least_val_loss = val_loss

    # Update ckpts file name
    checkpoint_filename = create_checkpoint_filename(args)
    checkpoint_path = os.path.join(args.ckpt_path, checkpoint_filename)

    # Model Training and Validation
    train_losses, train_accuracies, val_losses, val_accuracies = training.train(model,
                                                                                train_loader=train_loader,
                                                                                validation_loader=val_loader,
                                                                                criterion=criterion,
                                                                                optimizer=optimizer, epochs=args.epochs,
                                                                                writer=writer,
                                                                                checkpoint_path=checkpoint_path,
                                                                                current_epoch=current_epoch,
                                                                                least_val_loss=least_val_loss,
                                                                                args=args)
    # Plot losses and accuracies
    plot_losses(train_losses=train_losses, val_losses=val_losses, file_name=f"{args.model.lower()}-loss.png")
    plot_accuracies(train_accuracies, val_accuracies, file_name=f"{args.model.lower()}-acc.png")

    # Test and evaluation and confusion matrix
    import model_utils.multi_class_image_classification.evaluation as evaluation
    criterion = nn.CrossEntropyLoss()
    test_accuracy, test_loss, confusion = evaluation.evaluate(model, test_loader, criterion=criterion, args= args)
    print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")
    print("Confusion Matrix:")
    print(confusion)