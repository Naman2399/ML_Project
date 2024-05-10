import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, update_frequency=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

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
            loss = criterion(outputs, torch.argmax(labels, 1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == torch.argmax(labels, 1)).sum().item()
            total_train += labels.size(0)

            train_iterator.set_postfix({'train_loss': running_loss / total_train, 'train_acc': correct_train / total_train})

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
                val_loss += criterion(outputs, torch.argmax(labels, 1)).item()

                # Compute validation accuracy
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == torch.argmax(labels, 1)).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / total_val
        val_losses.append(avg_val_loss)

        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        train_iterator.set_postfix({'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_acc': val_accuracy, 'train_acc': train_accuracy})
        train_iterator.close()

    return train_losses, val_losses, train_accuracies, val_accuracies

def test_model(model, dataloader, num_classes):
    y_true = []
    y_pred = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))  # Use argmax to get predicted class index
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class index
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(np.argmax(y_true, axis=1), y_pred)
    precision = precision_score(np.argmax(y_true, axis=1), y_pred, average='weighted')
    recall = recall_score(np.argmax(y_true, axis=1), y_pred, average='weighted')
    f1 = f1_score(np.argmax(y_true, axis=1), y_pred, average='weighted')
    roc_auc = roc_auc_score(np.argmax(y_true, axis=1), np.eye(num_classes)[y_pred], multi_class='ovr')
    average_loss = total_loss / total_samples

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'ROC-AUC Score: {roc_auc:.2f}')
    print(f'Average Loss: {average_loss:.2f}')

    return accuracy, precision, recall, f1, roc_auc, average_loss