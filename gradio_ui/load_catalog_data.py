from unitxt.utils import load_json
from unitxt.file_utils import get_all_files_in_dir
import os
import constants as cons



def get_templates(template_data):
        def get_from_str(template_str):
            if template_data.endswith('.all'):
                subfolders = template_str.split('.')[:-1]+['all.json']
                template_file = os.path.join(*([cons.CATALOG_DIR]+subfolders))
                return set(load_json(template_file)['items'])
            else:
                return {template_str}

        if isinstance(template_data,str):
            return get_from_str(template_data)
        else:
            templates = set()
            for item in template_data:
                templates.update(get_from_str(item))
            return templates

def load_cards_data():
    def is_valid_data(data):
        for item in ['task','templates']:
            if isinstance(data[item],dict):
                return False
        return True

    cards_data = dict()
    cards_dir = os.path.join(cons.CATALOG_DIR,'cards')
    cards = get_all_files_in_dir(cards_dir,recursive=True)
    for card in cards:
        data = load_json(card)
        if not is_valid_data(data):
            continue
        dataset = f"cards.{os.path.basename(card)}".replace('.json','')        
        task = data['task']
        templates = get_templates(data['templates'])
        cards_data.setdefault(task,{}).update({dataset:templates})
    return cards_data


def get_catalog_items(items_type):
    items = []
    items_dir = os.path.join(cons.CATALOG_DIR,items_type)
    files = get_all_files_in_dir(items_dir,recursive=True)    
    for file in files:
        key = file.split(os.sep)
        start_index = key.index(items_type)
        key = key[start_index:]
        key = '.'.join(key).replace('.json','')
        items.append(key)
    return items

print(get_catalog_items('formats'))