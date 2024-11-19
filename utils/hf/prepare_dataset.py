import glob
import os
import shutil
import tempfile

from huggingface_hub import HfApi

files = glob.glob("./src/unitxt/*.py")
files.append("README.md")

api = HfApi(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

print("Uploading files from src/unitxt/ to hf:unitxt/data")

with tempfile.TemporaryDirectory() as temp_dir:
    for file in files:
        file_name = os.path.basename(file)

        if file_name == "__init__.py":
            continue

        if file_name == "dataset.py":
            file_name = "data.py"

        shutil.copy(file, os.path.join(temp_dir, file_name))

        print(f"  - {file_name}")

    api.upload_folder(
        folder_path=temp_dir,
        delete_patterns="*.py",  # delete any unused python files
        repo_id="unitxt/data",
        repo_type="dataset",
    )

with open(".src/unitxt/version.py") as f:
    version = f.read().strip().replace("version = ", "").replace('"', "")

api.create_tag(repo_id="unitxt/data", repo_type="dataset", tag=version)
