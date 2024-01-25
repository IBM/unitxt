import os

from unitxt.file_utils import get_all_files_in_dir
from unitxt.ui.constants import AUGMENTABLE_STR, CATALOG_DIR
from unitxt.utils import load_json


def safe_load_json(file):
    try:
        json = load_json(file)
    except:
        json = {"Error in loading json"}
    return json


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
        templates = get_from_str(template_data)
    else:
        templates = set()
        for item in template_data:
            templates.update(get_from_str(item))
    # templates.add("templates.key_val_with_new_lines")
    templates_jsons = {
        template: safe_load_json(get_file_from_item_name(template, dir))
        for template in templates
    }
    return templates, templates_jsons


def load_cards_data():
    def is_valid_data(data):
        for item in ["task", "templates"]:
            if isinstance(data[item], dict):
                return False
        return True

    cards_data = {}
    json_data = {}
    catalog_dirs = get_catalog_dirs()
    for dir in catalog_dirs:
        cards, card_jsons = get_catalog_items_from_dir("cards", dir)
        json_data.update(card_jsons)
        for card in cards:
            card_file = get_file_from_item_name(card, dir)
            data = load_json(card_file)
            if not is_valid_data(data):
                continue
            task = data["task"]
            if task not in cards_data:
                is_augmentable = check_augmentable(task, dir)
            else:
                is_augmentable = cards_data[task][AUGMENTABLE_STR]
            templates, templates_jsons = get_templates(data["templates"], dir)
            json_data.update(templates_jsons)
            cards_data.setdefault(task, {}).update(
                {card: templates, AUGMENTABLE_STR: is_augmentable}
            )
    formats, formats_jsons = get_catalog_items("formats")
    json_data.update(formats_jsons)
    instructions, instructiosn_jsons = get_catalog_items("instructions")
    json_data.update(instructiosn_jsons)
    return cards_data, json_data


def check_augmentable(task_name, dir):
    task_file = get_file_from_item_name(task_name, dir)
    task_data = load_json(task_file)
    return AUGMENTABLE_STR in task_data


def get_file_from_item_name(item_name, dir):
    file = os.path.join(dir, item_name.replace(".", os.sep) + ".json")
    if not os.path.exists(file):
        file = os.path.join(
            dir, item_name.replace("all", "").replace(".", os.sep), "json", "all.json"
        )
        if not os.path.exists(file):
            file = os.path.join(CATALOG_DIR, item_name.replace(".", os.sep) + ".json")
    return file


def get_catalog_items_from_dir(items_type, dir):
    items = []
    jsons = {}
    items_dir = os.path.join(dir, items_type)
    files = get_all_files_in_dir(items_dir, recursive=True)
    for file in files:
        if "DS_Store" in file:
            continue
        key = file.split(os.sep)
        start_index = key.index(items_type)
        key = key[start_index:]
        key = ".".join(key).replace(".json", "")
        items.append(key)
        jsons[key] = safe_load_json(file)
    return items, jsons


def get_catalog_items(items_type):
    items = []
    jsons = {}
    for dir in get_catalog_dirs():
        dir_items, dir_jsons = get_catalog_items_from_dir(items_type, dir)
        items.extend(dir_items)
        jsons.update(dir_jsons)
    return sorted(items), jsons


if __name__ == "__main__":
    data, jsons = load_cards_data()
    with open("cards_data.txt", "w") as file:
        for key, value in data.items():
            file.write(f"{key}: {value}\n")
    with open("jsons.txt", "w") as file:
        for key, value in jsons.items():
            file.write(f"{key}: {value}\n")
