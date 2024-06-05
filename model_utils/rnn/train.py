import torch
from tqdm import tqdm
from utils.checkpoints import *
from model_utils.rnn.validation import validation
import torch.nn as nn

# Training function
def train(model, dataset, criterion, optimizer, epochs, writer, checkpoint_path, current_epoch, least_val_loss, args, device):

    # Initializing list for  train, validation accuracy and losses
    train_losses = []
    val_losses = []

    pbar = tqdm(range(current_epoch, epochs), desc="Epochs")
    for epoch in pbar:

        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        # Training
        model.train()

        # Forward Propagation
        input, target = dataset.random_training_batch()
        target_one_hot = nn.functional.one_hot(target, num_classes= dataset.n_characters).to(torch.float).to(device)
        hidden = model.init_hidden(batch_size= args.batch)
        input = input.to(device)
        target = target.to(device)
        hidden = hidden.to(device)
        output_hat, hidden = model(input, hidden)

        # Backward Propagation
        optimizer.zero_grad()
        loss = criterion(target_one_hot, output_hat)
        train_loss = loss.item()/args.batch
        loss.backward()
        optimizer.step()

        # Progress Bar
        pbar.set_postfix({'Train Loss': train_loss})
        train_losses.append(train_loss)

        # Validation
        val_loss = validation(model, dataset, criterion, device, args)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Adding in Tensorboard
        writer.add_scalar('train_loss', train_loss, epoch + 1)
        writer.add_scalar('valid_loss', val_loss, epoch + 1)

        # Always saving best model with least validation loss
        # Saving model checkpoints
        if val_loss < least_val_loss :
            save_checkpoint(args, model, optimizer, epoch, checkpoint_path, val_loss)
            least_val_loss = val_loss

        # Stopping Criteria for model
        if len(val_losses) >= 10 and  stopping_criteria(val_losses[-10:], val_loss) :
            print("Early Stopping .....")
            break

    return train_losses, val_losses


def stopping_criteria(val_losses, val_loss, patience_level = 0.05) :

    if val_loss <= min(val_losses) + patience_level :
        return False
    return  True