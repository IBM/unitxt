import csv
import os
from collections import defaultdict

directory = 'ilab/ilab_results'
directory2 = 'ilab/ilab_results2'

files1 = os.listdir(directory)
files2 = os.listdir(directory2)

for file_name in os.listdir(directory2):
    if file_name not in files1:
        print(f'file {file_name} is only at 2')
    else:
        with open(os.path.join(directory, file_name), 'r') as file:
            content1 = file.read()
        with open(os.path.join(directory2, file_name), 'r') as file:
            content2 = file.read()
        if content1 == content2:
            print(f'file {file_name} is equal at both. deleting')
            #os.remove(os.path.join(directory, file_name))
        else:
            print(f'file {file_name} is DIFFERENT')
