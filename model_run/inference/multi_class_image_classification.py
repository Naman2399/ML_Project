import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import model_utils.multi_class_image_classification.train as training
from utils.checkpoints import create_checkpoint_filename, load_checkpoint
from utils.data_utils import plot_losses, plot_accuracies


def run(X, args, device, model, test_loader, writer, epochs_ckpts):

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

    epochs = list(epochs_ckpts.keys())
    epochs.sort()

    # Test and evaluation and confusion matrix
    import model_utils.multi_class_image_classification.evaluation as evaluation
    for epoch in epochs :
        model, _, _, _ = load_checkpoint(model, optimizer,checkpoint_path=epochs_ckpts[epoch])

        test_accuracy, test_loss, confusion = evaluation.evaluate(model, test_loader, criterion=criterion, args=args)
        print("-" * 20 , f"Epoch : {epoch}", "-" * 20)
        print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")
        print("Confusion Matrix:")
        print(confusion)
        writer.add_scalar('test_loss', test_loss, epoch + 1)
        writer.add_scalar('test_acc', test_accuracy, epoch + 1)





