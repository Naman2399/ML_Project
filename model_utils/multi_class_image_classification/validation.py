import torch
def validation(model, validation_loader, criterion, optimizer, epochs, args):

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()

    val_loss /= len(validation_loader)
    val_acc = 100 * correct / total

    return val_loss, val_acc
