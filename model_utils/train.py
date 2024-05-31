import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_utils.checkpoints import save_checkpoint
from model_utils.validation import validation


# Training function
def train(model, train_loader, validation_loader, criterion, optimizer, epochs, writer, checkpoint_path, current_epoch, args):

    # Initializing list for  train, validation accuracy and losses
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(current_epoch, epochs):

        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress Bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for inputs, labels in pbar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()

            # Adding details in progress bar as postfix
            pbar.set_postfix({'Train Loss': running_loss / len(train_loader), 'Train Accuracy': 100 * correct / total})

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = validation(model, validation_loader, criterion, optimizer, epochs)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"\n Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        writer.add_scalar('train_loss', train_loss, epoch + 1)
        writer.add_scalar('valid_loss', val_loss, epoch + 1)
        writer.add_scalar('train_acc', train_acc, epoch + 1)
        writer.add_scalar('valid_acc', val_acc, epoch + 1)

        # Saving model checkpoints
        save_checkpoint(args, model, optimizer, epoch, checkpoint_path)

    return train_losses, train_accuracies, val_losses, val_accuracies
