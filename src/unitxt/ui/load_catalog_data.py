import os

from ..file_utils import get_all_files_in_dir
from ..utils import load_json
from .settings import AUGMENTABLE_STR, CATALOG_DIR


def get_catalog_dirs():
    unitxt_dir = CATALOG_DIR
    private_dir = None
    if "UNITXT_ARTIFACTORIES" in os.environ:
        env_dirs = os.environ["UNITXT_ARTIFACTORIES"].split(":")
        if len(env_dirs) > 1:
            raise ValueError(
                f"Expecting a maximum of one catalog in addition to unitxt catalog, found {len(env_dirs)}: {env_dirs}"
            )
        private_dir = env_dirs[0]
    return unitxt_dir, private_dir


UNITXT_DIR, PRIVATE_DIR = get_catalog_dirs()


def safe_load_json(file):
    try:
        json = load_json(file)
    except:
        json = {"Error in loading json"}
    return json


def get_templates(template_data):
    def get_from_str(template_str):
        if template_str.endswith(".all"):
            template_file = get_file_from_item_name(template_str)
            return set(load_json(template_file)["items"])
        return {template_str}

    if isinstance(template_data, str):
        templates = get_from_str(template_data)
    else:
        templates = set()
        for item in template_data:
            if isinstance(item, str):
                templates.update(get_from_str(item))
    # templates.add("templates.key_val_with_new_lines")
    templates_jsons = {
        template: safe_load_json(get_file_from_item_name(template))
        for template in templates
    }
    return templates, templates_jsons


def load_cards_data():
    def is_valid_data(data):
        for item in ["task", "templates"]:
            if item not in data or isinstance(data[item], dict):
                return False
        return True

    cards_data = {}
    json_data = {}
    unitxt_card_jsons = get_catalog_items_from_dir("cards", UNITXT_DIR)
    private_card_jsons = {}
    if PRIVATE_DIR:
        private_card_jsons = get_catalog_items_from_dir("cards", PRIVATE_DIR)
    card_jsons = private_card_jsons
    for card in unitxt_card_jsons:
        if card not in card_jsons:
            card_jsons[card] = unitxt_card_jsons[card]

    cards = card_jsons.keys()
    json_data.update(card_jsons)
    for card in cards:
        data = card_jsons[card]
        if not is_valid_data(data):
            continue
        task = data["task"].split("[")[0]
        if task not in cards_data:
            is_augmentable = check_augmentable(task)
        else:
            is_augmentable = cards_data[task][AUGMENTABLE_STR]
        templates, templates_jsons = get_templates(data["templates"])
        json_data.update(templates_jsons)
        cards_data.setdefault(task, {}).update(
            {card: templates, AUGMENTABLE_STR: is_augmentable}
        )
    formats, formats_jsons = get_catalog_items("formats")
    json_data.update(formats_jsons)
    system_prompts, system_prompts_jsons = get_catalog_items("system_prompts")
    json_data.update(system_prompts_jsons)
    _, tasks_jsons = get_catalog_items("tasks")
    json_data.update(tasks_jsons)

    return cards_data, json_data, formats, system_prompts


def check_augmentable(task_name):
    task_file = get_file_from_item_name(task_name)
    task_data = load_json(task_file)
    return AUGMENTABLE_STR in task_data


def get_file_from_item_name(item_name):
    dirs = [PRIVATE_DIR, UNITXT_DIR] if PRIVATE_DIR else [UNITXT_DIR]
    for dir in dirs:
        file = os.path.join(dir, item_name.replace(".", os.sep) + ".json")
        if os.path.exists(file):
            return file
        file = os.path.join(
            dir, item_name.replace("all", "").replace(".", os.sep), "json", "all.json"
        )
        if os.path.exists(file):
            return file
    return os.path.join(CATALOG_DIR, item_name.replace(".", os.sep) + ".json")


def get_catalog_items_from_dir(items_type, dir):
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
        jsons[key] = safe_load_json(file)
    return jsons


def get_catalog_items(items_type):
    unitxt_items_to_jsons = get_catalog_items_from_dir(items_type, UNITXT_DIR)
    if not PRIVATE_DIR:
        items_to_jsons = unitxt_items_to_jsons
    else:
        items_to_jsons = get_catalog_items_from_dir(items_type, PRIVATE_DIR)
        for item in unitxt_items_to_jsons:
            if item not in items_to_jsons:
                items_to_jsons[item] = unitxt_items_to_jsons[item]
    items = items_to_jsons.keys()
    return sorted(items), items_to_jsons


if __name__ == "__main__":
    data, jsons, formats, system_prompts = load_cards_data()
    stuff = {
        "cards_data": data,
        "jsons": jsons,
        "formats": formats,
        "system_prompts": system_prompts,
    }
    for bla in stuff:
        with open(f"{bla}.txt", "w") as file:
            bla = stuff[bla]
            if isinstance(bla, dict):
                for key, value in bla.items():
                    file.write(f"{key}: {value}\n\n")
            else:
                for val in bla:
                    file.write(f"{val}\n\n")
