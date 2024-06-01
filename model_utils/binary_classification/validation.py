import torch
def validation(model, validation_loader, criterion, optimizer, epochs):

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:

            # Forward Pass
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()

            # Compute validation accuracy
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(validation_loader)
    val_acc = 100 * correct / total

    return val_loss, val_acc
