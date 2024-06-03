import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            encoded_feats, reconstructed_output = model(inputs)
            loss = criterion(reconstructed_output, inputs) # Here the inputs as same as that of labels
            total_loss += loss.item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)

    return avg_loss