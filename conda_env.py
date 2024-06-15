import chardet

pip_list_file = 'pip_list.txt'
conda_env_file = 'environment.yml'

# Detect the encoding of the pip_list.txt file
with open(pip_list_file, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

# Read the file with the detected encoding
try:
    with open(pip_list_file, 'r', encoding=encoding) as f:
        pip_packages = f.readlines()

    # Create environment.yml content
    conda_env_content = """
name: ml_project
dependencies:
  - python=3.12
  - pip
  - pip:
"""

    for package in pip_packages:
        conda_env_content += f"    - {package.strip()}\n"

    with open(conda_env_file, 'w', encoding='utf-8') as f:
        f.write(conda_env_content)

    print(f"Conda environment file '{conda_env_file}' created successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
