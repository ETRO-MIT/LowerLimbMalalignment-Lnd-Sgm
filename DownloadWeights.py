# Python packages
import os
from huggingface_hub import snapshot_download, hf_hub_download


# Set the working directory
main_directory = os.getcwd()

# Download weights
print('Downloading weights...')
snapshot_download(
    repo_id="samador7/sgm-lnd-det-v1",
    allow_patterns="*.pth",
    local_dir=main_directory
)

# Download README.md
print('Downloading README...')
hf_hub_download(
    repo_id="samador7/sgm-lnd-det-v1",
    filename="README.md",
    local_dir=os.path.join(main_directory, 'Weights')
)

# Download dictionary
print('Downloading dictionary...')
hf_hub_download(
    repo_id="samador7/sgm-lnd-det-v1",
    filename="Weights.py",
    local_dir=os.path.join(main_directory, 'Weights')
)
