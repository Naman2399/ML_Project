import torch
from torchvision import transforms

def resize_and_pad(image_tensor, target_size=(299, 299)):
  """
  Resizes and pads a batch of image tensors to the target size.

  Args:
      image_tensor: A PyTorch tensor of shape (batch_size, image_channel, image_height, image_width)
          representing a batch of images.
      target_size: A tuple (height, width) specifying the desired output size.

  Returns:
      A resized and padded PyTorch tensor of shape (batch_size, 3, target_size[0], target_size[1]).
  """

  # Create a resize transform
  resize_transform = transforms.Resize(target_size)

  # Apply the transform to each image in the batch
  resized_image_tensor = resize_transform(image_tensor)

  # Pad with zeros to achieve the target size (optional)
  padding = [0, (target_size[1] - image_tensor.shape[3]) // 2,  # Left/right padding
            0, (target_size[0] - image_tensor.shape[2]) // 2]  # Top/bottom padding
  pad_transform = transforms.Pad(padding, padding_mode='constant')

  # Optionally, apply padding
  if resized_image_tensor.shape[3] != target_size[1] or resized_image_tensor.shape[2] != target_size[0]:
    padded_image_tensor = pad_transform(resized_image_tensor)
  else:
    padded_image_tensor = resized_image_tensor

  return padded_image_tensor


if __name__ == "__main__" :
    # Example usage
    # image_tensor = torch.randn(2, 3, 211, 274)  # Example batch of 2 images
    # resized_padded_tensor = resize_and_pad(image_tensor)
    #
    # print(resized_padded_tensor.shape)  # Output: torch.Size([2, 3, 299, 299])
    exit()