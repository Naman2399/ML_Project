import torch
from tqdm import tqdm

from utils.checkpoints import save_checkpoint
from model_utils.multi_class_image_classification.validation import validation


# Training function
def train(model, train_loader, validation_loader, criterion, optimizer, epochs, writer, checkpoint_path, current_epoch, least_val_loss, args):

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

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Computing training accuracy
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()

            # Adding details in progress bar as postfix
            pbar.set_postfix({'Train Loss': running_loss / total, 'Train Accuracy': 100 * correct / total})

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = validation(model, validation_loader, criterion, optimizer, epochs, args)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        writer.add_scalar('train_loss', train_loss, epoch + 1)
        writer.add_scalar('valid_loss', val_loss, epoch + 1)
        writer.add_scalar('train_acc', train_acc, epoch + 1)
        writer.add_scalar('valid_acc', val_acc, epoch + 1)

        # Always saving best model with least validation loss
        # Saving model checkpoints
        if val_loss < least_val_loss :
            save_checkpoint(args, model, optimizer, epoch, checkpoint_path, val_loss)
            least_val_loss = val_loss

        # Stopping Criteria for model
        if len(val_losses) >= 10 and  stopping_criteria(val_losses[-10:], val_loss) :
            print("Early Stopping .....")
            break

    return train_losses, train_accuracies, val_losses, val_accuracies


def stopping_criteria(val_losses, val_loss, patience_level = 0.05) :

    if val_loss <= min(val_losses) + patience_level :
        return False
    return  True