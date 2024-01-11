import glob
import os

from huggingface_hub import HfApi

from src.unitxt.logging_utils import get_logger

logger = get_logger()

files = glob.glob("./src/unitxt/*.py")

api = HfApi(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

logger.info("Uploading files from src/unitxt/ to hf:unitxt/data")

for file in files:
    file_name = os.path.basename(file)

    if file_name == "__init__.py":
        continue

    if file_name == "dataset.py":
        file_name = "data.py"

    logger.info(f"  - {file_name}")

    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file_name,
        repo_id="unitxt/data",
        repo_type="dataset",
    )
