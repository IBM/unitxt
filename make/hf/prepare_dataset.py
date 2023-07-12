from huggingface_hub import HfApi
import glob
import os

files = glob.glob("./src/unitxt/*.py")

api = HfApi()

print('Uploading files from src/unitxt/ to hf:unitxt/data')

for file in files:
    
    file_name = os.path.basename(file)
    
    if file_name == '__init__.py':
        continue
    
    if file_name == 'dataset.py':
        file_name = 'data.py'
    
    print(f'  - {file_name}')
    
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file_name,
        repo_id="unitxt/data",
        repo_type="dataset",
    )