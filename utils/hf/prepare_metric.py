import glob
import os
import shutil
import tempfile

from huggingface_hub import HfApi

files = glob.glob("./src/unitxt/*.py")

api = HfApi(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

print("\nUploading files from src/unitxt/ to hf:unitxt/metric")

with tempfile.TemporaryDirectory() as temp_dir:
    for file in files:
        file_name = os.path.basename(file)

        if file_name == "__init__.py":
            continue

        shutil.copy(file, os.path.join(temp_dir, file_name))

        print(f"  - {file_name}")

    api.upload_folder(
        folder_path=temp_dir,
        delete_patterns="*.py",  # delete any unused python files
        repo_id="unitxt/metric",
        repo_type="space",
        run_as_future=True,
    )
