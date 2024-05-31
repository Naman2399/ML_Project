import torch

def get_available_device():
  """
  Checks if a CUDA device is available, otherwise returns 'cpu'.

  Returns:
      str: 'cuda:0' if a CUDA device is available, otherwise 'cpu'.
  """
  return 'cuda:0' if torch.cuda.is_available() else 'cpu'
