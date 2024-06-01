import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()

            all_predictions.extend(predicted.tolist())
            all_targets.extend(true_labels.tolist())

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)

    confusion = confusion_matrix(all_targets, all_predictions)

    return accuracy, avg_loss, confusion