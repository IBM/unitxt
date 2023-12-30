import os


def get_directory_size(directory):
    total = 0
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path):
            total += os.path.getsize(full_path)
        elif os.path.isdir(full_path):
            total += get_directory_size(full_path)
    return total
