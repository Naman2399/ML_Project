import torch
import models.lenet_5 as models

def view_layer_weights(model_path):
  """
  Prints information about layer-wise weights in a state dictionary.

  Args:
      model_state_dict: PyTorch model state dictionary containing layer weights.
  """

  # Load model state dictionary
  dict = torch.load(model_path)
  model_state_dict = dict['model_state_dict']

  for name, param in model_state_dict.items():
    if 'weight'  in name or 'bias' in name :  # Filter for weight parameters (not biases)
      print(f"Layer Name: {name} ---- Shape {param.shape}")


# Load Weights for block prefix
def model_load_block_prefix_weights(model, checkpoint_path, block_prefix) :

  dict = torch.load(checkpoint_path)
  model_state_dict = dict['model_state_dict']

  # All the keys for current block_prefix
  prefix_keys = [key for key in model_state_dict.keys() if key.startswith(f'{block_prefix}.')]
  # Create a new dictionary with only branch1 weights
  prefix_state_dict = {k: model_state_dict[k] for k in prefix_keys}
  # Load weights for block prefix
  model.load_state_dict(prefix_state_dict, strict=False)
  print(f"Loaded weights for {block_prefix} from: {checkpoint_path}")
  return model

def model_freeze_block_layers(model, block_name) :
  module = model.get_submodule(block_name)
  module.requires_grad_(False)
  return model

def check_layer_grads(model):
  """
  Prints information about trainable and frozen layers in a PyTorch model.
  """

  trainable_layers = []
  frozen_layers = []

  for name, param in model.named_parameters():
    if param.requires_grad:
      trainable_layers.append(name)
    else:
      frozen_layers.append(name)

  # Print results
  if trainable_layers:
    print("Trainable Layers:")
    for layer in trainable_layers:
      print(f"\t- {layer}")
  else:
    print("No trainable layers found.")

  if frozen_layers:
    print("Frozen Layers:")
    for layer in frozen_layers:
      print(f"\t- {layer}")
  else:
    print("No frozen layers found.")


def get_layer_by_name(model, layer_name):
  """
  Attempts to retrieve a layer from the model by its name.

  Args:
      model: PyTorch model instance.
      layer_name: Name of the layer to retrieve.

  Returns:
      nn.Module: The retrieved layer or None if not found.
  """

  for name, module in model.named_modules():
    if name == layer_name:
      return module
  return None


'''
Driver Code
'''
# Example usage (assuming state_dict is loaded from a checkpoint)
# view_layer_weights('ckpts/lenetv2_cifar10_lenetv2_lr_0.0005.pt')
# model = model_load_block_prefix_weights(models.LeNet5_v2(), 'ckpts/lenetv2_cifar10_lenetv2_lr_0.0005.pt', 'conv_block')
# model = model_freeze_block_layers(model, 'conv_block')
# check_layer_grads(model)
