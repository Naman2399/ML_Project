import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # One fully connected layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


# Training Function
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, update_frequency=20):
    criterion = nn.BCELoss()
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
def test_model(model, dataloader):
    y_true = []
    y_pred = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.BCELoss()


    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    average_loss = total_loss / total_samples

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'ROC-AUC Score: {roc_auc:.2f}')
    print(f'Average Loss: {average_loss:.2f}')

    return accuracy, precision, recall, f1, roc_auc, average_loss
