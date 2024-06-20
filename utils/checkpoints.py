import os
import torch
import re

def epoch_completed(args):
  '''

  Args:
    args: args

  Returns:
    next epoch number to start training

  '''
  try :
    # Check the epochs which are already computed
    file_name = create_checkpoint_filename(args)
    dir_path = os.path.join(args.ckpt_path, file_name[:-3])
    print(os.listdir(dir_path))
    final_epoch_num = -1
    ckpt_file_name = None
    for file_name in os.listdir(dir_path):
      match = re.search(r"epoch_(\d+)\.pt", file_name)
      epoch_num = int(match.group(1))
      if epoch_num > final_epoch_num:
        final_epoch_num = epoch_num
        ckpt_file_name = os.path.join(dir_path, str(file_name))
  except :
    final_epoch_num = 0
    ckpt_file_name = None 

  return final_epoch_num, ckpt_file_name

def create_checkpoint_filename(args):
  """
  Creates a checkpoint filename based on arguments.

  Args:
      args: Namespace object containing parsed arguments.

  Returns:
      str: The generated checkpoint filename.
  """

  filename = f"{args.exp_name}_{args.dataset}_{args.model}_lr_{args.lr}_bs_{args.batch}_epochs_{args.epochs}.pt"
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

  os.makedirs(checkpoint_path[:-3], exist_ok=True)  # Create directory if it doesn't exist
  ckpt_path = os.path.join(checkpoint_path[:-3], f"epoch_{epoch}.pt")
  # Save model weights and optimizer state
  torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'val_loss' : val_loss,
  }, ckpt_path)

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


