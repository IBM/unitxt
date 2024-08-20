import os
from collections import defaultdict

directory = 'ilab/ilab_results'
names = ['cat', 'clapnq', 'finqa', 'entities_all', 'watson_emotion_classes_first']
experiments = ['0_shots', '5_shots', 'yaml']
models = ['base', 'train']

files = os.listdir(directory)

missing_counter = 0
for name in names:
    for model in models:
        for experiment in experiments:
            file_prefix = f"{name}_{model}_{experiment}"
            found = False
            for file in files:
                if file.startswith(file_prefix):
                    found = True
            if not found:
                print(f'Missing: {name} {model} {experiment}')
                missing_counter +=1

print(f'Missing {missing_counter} / {len(names) * len(experiments) * len(models)}')
