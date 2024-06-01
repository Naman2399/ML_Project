import torch

def validation(model, val_loader, criterion) :
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(val_loader)
    return val_loss
