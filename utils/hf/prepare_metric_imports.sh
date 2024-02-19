#!/bin/bash
# Set the directory path
dir_path=src/unitxt
target_file=metric
# Loop over each .py file in the directory
for file in $dir_path/*.py; do
    # Extract the filename without extension
    file_name=$(basename -- "$file")
    file_name="${file_name%.*}"

    # Skip the dataset.py and __init__.py files
    if [[ "$file_name" != "$target_file" && "$file_name" != "__init__" && "$file_name" != "metric" && "$file_name" != "dataset"  && "$file_name" != "api" ]]
    then
        # Add the import statement to dataset.py using isort
        isort -a "from .$file_name import __file__ as _" $dir_path/$target_file.py
    fi
done
