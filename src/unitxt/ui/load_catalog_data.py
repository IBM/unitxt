import os

from unitxt.file_utils import get_all_files_in_dir
from unitxt.ui.constants import AUGMENTABLE_STR, CATALOG_DIR
from unitxt.utils import load_json


def get_catalog_dirs():
    dirs = [CATALOG_DIR]
    if "UNITXT_ARTIFACTORIES" in os.environ:
        env_dirs = os.environ["UNITXT_ARTIFACTORIES"]
        dirs.extend(env_dirs.split(":"))
    return dirs


def get_templates(template_data, dir):
    def get_from_str(template_str):
        if template_data.endswith(".all"):
            template_file = get_file_from_item_name(template_str, dir)
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
    catalog_dirs = get_catalog_dirs()
    for dir in catalog_dirs:
        cards = get_catalog_items_from_dir("cards", dir)
        for card in cards:
            data = load_json(get_file_from_item_name(card, dir))
            if not is_valid_data(data):
                continue
            task = data["task"]
            is_augmentable = check_augmentable(task, dir)
            templates = get_templates(data["templates"], dir)
            cards_data.setdefault(task, {}).update(
                {card: templates, AUGMENTABLE_STR: is_augmentable}
            )
    return cards_data


def check_augmentable(task_name, dir):
    task_file = get_file_from_item_name(task_name, dir)
    task_data = load_json(task_file)
    return AUGMENTABLE_STR in task_data


def get_file_from_item_name(item_name, dir):
    return os.path.join(dir, item_name.replace(".", os.sep) + ".json")


def get_catalog_items_from_dir(items_type, dir):
    items = []
    items_dir = os.path.join(dir, items_type)
    files = get_all_files_in_dir(items_dir, recursive=True)
    for file in files:
        key = file.split(os.sep)
        start_index = key.index(items_type)
        key = key[start_index:]
        key = ".".join(key).replace(".json", "")
        items.append(key)
    return items


def get_catalog_items(items_type):
    items = []
    for dir in get_catalog_dirs():
        items.extend(get_catalog_items_from_dir(items_type, dir))
    return items


if __name__ == "__main__":
    load_cards_data()
