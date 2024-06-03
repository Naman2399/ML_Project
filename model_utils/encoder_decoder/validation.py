import torch

def validation(model, validation_loader, criterion):

    # Validation

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            encoded_feats, reconstructed_output = model(inputs)
            loss = criterion(reconstructed_output, inputs) # Here labels are same as that of inputs
            val_loss += loss.item()
            total += labels.size(0)

    val_loss /= len(validation_loader)

    return val_loss
