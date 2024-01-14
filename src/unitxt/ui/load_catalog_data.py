import os

from ..file_utils import get_all_files_in_dir
from ..utils import load_json
from .constants import AUGMENTABLE_STR, CATALOG_DIR


def get_templates(template_data):
    def get_from_str(template_str):
        if template_data.endswith(".all"):
            subfolders = template_str.split(".")[:-1] + ["all.json"]
            template_file = os.path.join(*([CATALOG_DIR, *subfolders]))
            return set(load_json(template_file)["items"])

        return {template_str}

    if isinstance(template_data, str):
        return get_from_str(template_data)

    templates = set()
    for item in template_data:
        templates.update(get_from_str(item))
    return templates


def load_cards_data():
    def is_valid_data(data):
        for item in ["task", "templates"]:
            if isinstance(data[item], dict):
                return False
        return True

    cards_data = {}
    cards = get_catalog_items("cards")
    for card in cards:
        data = load_json(get_file_from_item_name(card))
        if not is_valid_data(data):
            continue
        task = data["task"]
        is_augmentable = check_augmentable(task)
        templates = get_templates(data["templates"])
        cards_data.setdefault(task, {}).update(
            {card: templates, AUGMENTABLE_STR: is_augmentable}
        )
    return cards_data


def check_augmentable(task_name):
    task_file = get_file_from_item_name(task_name)
    task_data = load_json(task_file)
    return AUGMENTABLE_STR in task_data


def get_file_from_item_name(item_name):
    return os.path.join(CATALOG_DIR, item_name.replace(".", os.sep) + ".json")


def get_catalog_items(items_type):
    items = []
    items_dir = os.path.join(CATALOG_DIR, items_type)
    files = get_all_files_in_dir(items_dir, recursive=True)
    for file in files:
        key = file.split(os.sep)
        start_index = key.index(items_type)
        key = key[start_index:]
        key = ".".join(key).replace(".json", "")
        items.append(key)
    return items


if __name__ == "__main__":
    load_cards_data()
