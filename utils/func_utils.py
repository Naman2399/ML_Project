import os

def remove_folder_content(folder_path):
  """
  Removes the content of a folder but keeps the folder itself.

  Args:
      folder_path (str): Path to the folder.

  Raises:
      OSError: If an error occurs while removing a file or directory.
  """

  if os.path.exists(folder_path):
    # Check if path is a directory
    if not os.path.isdir(folder_path):
      print(f"Path '{folder_path}' is not a directory.")
      return

    for filename in os.listdir(folder_path):
      file_path = os.path.join(folder_path, filename)
      try:
        # Check if it's a file
        if os.path.isfile(file_path):
          os.remove(file_path)
          print(f"Removed file: {file_path}")
      except OSError as e:
        print(f"Error removing file/folder: {file_path} ({e})")


  else:
    print(f"Folder '{folder_path}' does not exist.")

