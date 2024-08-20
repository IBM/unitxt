import csv
import os
from collections import defaultdict

directory = 'ilab/ilab_results'

for file_name in os.listdir(directory):
    if file_name.endswith('run.csv'):
        print(file_name)
        with open(os.path.join(directory,file_name)) as file:
            model = ''
            if 'base' in file_name:
                model = 'base'
            elif 'trained' in file_name:
                model = 'trained'
            reader = csv.reader(file)
            headers = next(reader)
            row = next(reader)
            dict ={k: v for k,v in zip(headers, row)}
            print(dict['model_name'])

