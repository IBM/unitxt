import glob
import logging
import os

from huggingface_hub import HfApi

files = glob.glob("./src/unitxt/*.py")

api = HfApi(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

logging.info("\nUploading files from src/unitxt/ to hf:unitxt/metric")

for file in files:
    file_name = os.path.basename(file)

    if file_name == "__init__.py":
        continue

    logging.info(f"  - {file_name}")

    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file_name,
        repo_id="unitxt/metric",
        repo_type="space",
    )
