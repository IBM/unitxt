import glob
import os
import shutil
import tempfile

from huggingface_hub import HfApi

files = glob.glob("./src/unitxt/*.py")
files.append("README.md")

api = HfApi(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

print("\nUploading files from src/unitxt/ to hf:unitxt/metric")

with tempfile.TemporaryDirectory() as temp_dir:
    for file in files:
        file_name = os.path.basename(file)

        if file_name == "__init__.py":
            continue

        shutil.copy(file, os.path.join(temp_dir, file_name))

        if file == "README.md":
            header = """
---
title: "Unitxt Metric"
emoji: 📈
colorFrom: pink
colorTo: purple
sdk: static
app_file: README.md
pinned: false
---
            """.strip()

            # Step 1: Read the existing content
            with open(os.path.join(temp_dir, file_name)) as file:
                readme = file.read()

                readme = header + "\n" + readme

            with open(os.path.join(temp_dir, file_name), "w") as file:
                file.write(readme)

        print(f"  - {file_name}")

    api.upload_folder(
        folder_path=temp_dir,
        delete_patterns="*.py",  # delete any unused python files
        repo_id="unitxt/metric",
        repo_type="space",
    )

with open("./src/unitxt/version.py") as f:
    version = f.read().strip().replace("version = ", "").replace('"', "")

api.create_tag(repo_id="unitxt/metric", repo_type="space", tag=version)
