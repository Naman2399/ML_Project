import os
import torch
def create_checkpoint_filename(args):
  """
  Creates a checkpoint filename based on arguments.

  Args:
      args: Namespace object containing parsed arguments.

  Returns:
      str: The generated checkpoint filename.
  """

  filename = f"{args.exp_name}_{args.dataset}_{args.model}_lr_{args.lr}.pt"
  return filename

def save_checkpoint(args, model, optimizer, epoch, checkpoint_path, val_loss):
  """
  Saves model weights, optimizer state, and current epoch to a checkpoint file.

  Args:
      args: Namespace object containing parsed arguments.
      model: PyTorch model to save.
      optimizer: Optimizer object used for training.
      epoch: Current training epoch.
  """

  # Ensure checkpoint directory exists
  os.makedirs(args.ckpt_path, exist_ok=True)  # Create directory if it doesn't exist

  # Save model weights and optimizer state
  torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'val_loss' : val_loss,
  }, checkpoint_path)

  print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
  """
  Loads model weights, optimizer state, and current epoch from a checkpoint file.

  Args:
      model: PyTorch model to load weights into.
      optimizer: Optimizer object to load state into.
      checkpoint_path: Path to the checkpoint file.

  Returns:
      int: Current training epoch loaded from the checkpoint.
  """

  # Load checkpoint
  checkpoint = torch.load(checkpoint_path)

  # Load model weights
  model.load_state_dict(checkpoint['model_state_dict'])

  # Load optimizer state (if optimizer is not None)
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  # Load current epoch
  current_epoch = checkpoint['epoch']

  # Loading validation loss
  val_loss = checkpoint['val_loss']

  print(f"Checkpoint loaded: {checkpoint_path}")

  return model, optimizer, current_epoch, val_loss
