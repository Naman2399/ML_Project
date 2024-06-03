import torch
from tqdm import tqdm
from utils.checkpoints import *
from model_utils.encoder_decoder import validation

# Training function
def train(model, train_loader, validation_loader, criterion, optimizer, epochs, writer, checkpoint_path, current_epoch, least_val_loss, args):

    # Initializing list for  train, validation accuracy and losses
    train_losses = []
    val_losses = []
    weights = []
    for epoch in range(current_epoch, epochs):

        # Training
        model.train()
        running_loss = 0.0
        total = 0

        # Progress Bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for inputs, labels in pbar:

            # Fwd Propagation
            encoded_feats, reconstructed_output = model(inputs)
            loss = criterion(reconstructed_output, inputs) # As here label are same as that of inputs

            # Backward Propagation Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Perform Model Weight Tying
            model.weight_typing()

            running_loss += loss.item() # Adding loss
            total += labels.size(0)

            # Adding details in progress bar as postfix
            pbar.set_postfix({'Train Loss': running_loss / total})

        pbar.close()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss = validation.validation(model, validation_loader, criterion)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f},  Val Loss: {val_loss:.4f}")
        writer.add_scalar('train_loss', train_loss, epoch + 1)
        writer.add_scalar('valid_loss', val_loss, epoch + 1)

        # Get model weights
        if epoch % 50 == 0:
            weights.append(torch.reshape(model.get_encoder_weights(), (-1, 1)).squeeze())


        # Always saving best model with least validation loss
        # Saving model checkpoints
        if val_loss < least_val_loss :
            save_checkpoint(args, model, optimizer, epoch, checkpoint_path, val_loss)
            least_val_loss = val_loss

        # Stopping Criteria for model
        if len(val_losses) >= 10 and  stopping_criteria(val_losses[-10:], val_loss) :
            print("Early Stopping .....")
            break

    weights = torch.stack(weights)

    return train_losses, val_losses, weights


def stopping_criteria(val_losses, val_loss, patience_level = 0.05) :

    if val_loss <= min(val_losses) + patience_level :
        return False
    return  True