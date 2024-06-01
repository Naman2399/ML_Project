from tqdm import tqdm

from model_utils.linear_regression.validation import validation


# Define training function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, update_frequency=20):

    train_losses = []  # List to store training losses
    val_losses = []  # List to store validation losses

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        total_train = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_train += inputs.size(0)
            progress_bar.set_postfix({'train_loss': train_loss / total_train})
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loss = validation(model, val_loader, criterion)
        val_losses.append(val_loss)


        # Update progress bar description with validation loss
        if epoch % update_frequency == 0:
            progress_bar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})
        progress_bar.close()

    return train_losses, val_losses


