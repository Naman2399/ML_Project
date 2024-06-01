import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Training Function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, update_frequency=20):

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_accuracy = 0  # Initialize train_accuracy outside the inner loop

        train_iterator = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for i, (inputs, labels) in enumerate(train_iterator, 0):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute training accuracy
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            train_iterator.set_postfix({'train_loss': running_loss/total_train,  'train_acc': correct_train/total_train})


        avg_train_loss = running_loss / total_train
        train_losses.append(avg_train_loss)

        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation loss and accuracy
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

                # Compute validation accuracy
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / total_val
        val_losses.append(avg_val_loss)

        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        train_iterator.set_postfix({'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_acc': val_accuracy, 'train_acc': train_accuracy})
        train_iterator.close()
    return train_losses, val_losses, train_accuracies, val_accuracies


