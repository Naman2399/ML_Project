import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from utils.checkpoints import create_checkpoint_filename, load_checkpoint
from utils.data_utils import plot_losses, plot_accuracies, plot_weights
from model_utils.encoder_decoder.train import *
from model_utils.encoder_decoder.evaluation import *


def run(X, args, device, model, test_loader, train_loader, val_loader, writer) :


    model.to(device)
    # Tying model weights
    model.weight_typing()

    # Model Summary
    summary(model, (X.shape[1],))

    # Tensorboard - adding model
    input_tensor_example = torch.rand(((X.shape[1])))
    input_tensor_example = input_tensor_example.to(device)
    print("Input Tensor Example : ", input_tensor_example.shape)
    writer.add_graph(model, input_tensor_example)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    current_epoch = 0
    least_val_loss = float('inf')


    # Checkpoint filename
    if args.ckpt_filename is None:
        # Create new ckpts file
        checkpoint_filename = create_checkpoint_filename(args)
        checkpoint_path = os.path.join(args.ckpt_path, checkpoint_filename)

    else:
        print("Loading contents from checkpoints")
        # Load model, optimizer, epoch from checkpoints
        checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_filename)
        model, optimizer, current_epoch, val_loss = load_checkpoint(model, optimizer, checkpoint_path=checkpoint_path)
        current_epoch += 1
        least_val_loss = val_loss

    # Model Training and Validation
    train_losses, val_losses, weights = train(model,
                                    train_loader=train_loader, validation_loader=val_loader,
                                    criterion=criterion, optimizer=optimizer, epochs=args.epochs,
                                    writer=writer, checkpoint_path=checkpoint_path,
                                    current_epoch=current_epoch, least_val_loss=least_val_loss,
                                    args=args)

    # Plot losses and weights for encoder decoder model at different time step
    plot_losses(train_losses=train_losses, val_losses=val_losses, file_name=f"{args.model.lower()}-loss.png")
    plot_weights(weights= weights, file_name= checkpoint_filename[:-3] + ".png")

    # Test and evaluation and confusion matrix
    test_loss = evaluate(model, test_loader, criterion=criterion)
    print(f"Test Loss: {test_loss:.4f}")

    return