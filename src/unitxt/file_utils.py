import os
from typing import Optional


def get_all_files_in_dir(
    dir_path: str, recursive: bool = False, file_extension: Optional[str] = None
):
    """Get all files in a directory. Optionally recursively.

    Optionally filter by file extension.

    :param dir_path: The directory path to search for files.
    :param recursive: Whether to search recursively in subdirectories.
    :param file_extension: The file extension to filter by (e.g., '.txt').
    :return: A list of file paths.
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a directory")

    files = []
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if file_extension:
                if filename.endswith(file_extension):
                    files.append(os.path.join(root, filename))
            else:
                files.append(os.path.join(root, filename))
        if not recursive:
            break
    return files
