import atexit
import json
import logging
import os
import shutil

from ..error_utils import Documentation, UnitxtError, UnitxtWarning
from ..file_utils import get_all_files_in_dir
from ..utils import load_json
from .settings import AUGMENTABLE_STR, CATALOG_DIR

logger = logging.getLogger(__name__)


def get_catalog_dirs():
    unitxt_dir = CATALOG_DIR
    private_dir = None
    if "UNITXT_CATALOGS" in os.environ and "UNITXT_ARTIFACTORIES" == os.environ:
        raise UnitxtError(
            "Both UNITXT_CATALOGS and UNITXT_ARTIFACTORIES are set.  Use only UNITXT_CATALOG.  UNITXT_ARTIFICATORES is deprecated",
            Documentation.CATALOG,
        )

    env_dirs = []
    if "UNITXT_ARTIFACTORIES" == os.environ:
        UnitxtWarning(
            "UNITXT_ARTIFACTORIES is set but is deprecated, use UNITXT_CATALOGS instead.",
            Documentation.CATALOG,
        )
        env_dirs = os.environ["UNITXT_ARTIFACTORIES"].split(":")

    if "UNITXT_CATALOGS" in os.environ:
        env_dirs = os.environ["UNITXT_CATALOGS"].split(":")

    if len(env_dirs) > 1:
        raise ValueError(
            f"Expecting a maximum of one catalog in addition to unitxt catalog, found {len(env_dirs)}: {env_dirs}"
        )
    if len(env_dirs) > 0:
        private_dir = env_dirs[0]
    return unitxt_dir, private_dir


UNITXT_DIR, PRIVATE_DIR = get_catalog_dirs()


def safe_load_json(file):
    try:
        json = load_json(file)
    except:
        json = {"Error in loading json"}
    return json


def get_templates(template_data, card):
    if isinstance(template_data, str):
        return get_templates_from_str(template_data)

    if isinstance(template_data, dict):
        return get_templates_from_dict(template_data, card)

    if isinstance(template_data, list):
        return get_templates_from_list(template_data, card)

    raise TypeError(f"Unsupported template_data type: {type(template_data)}")


def get_templates_from_str(template_data):
    if template_data.endswith(".all"):
        template_file = get_file_from_item_name(template_data)
        templates = set(load_json(template_file)["items"])
    else:
        templates = set(template_data)

    templates_jsons = {
        template: safe_load_json(get_file_from_item_name(template))
        for template in templates
    }
    return templates, templates_jsons


def get_templates_from_dict(template_data, card):
    templates, templates_jsons = set(), {}
    templates_type = template_data.get("__type__")

    if templates_type == "templates_dict":
        _handle_templates_dict(template_data["items"], card, templates, templates_jsons)

    elif templates_type == "templates_list":
        _handle_templates_list(template_data["items"], card, templates, templates_jsons)

    else:
        _handle_plain_dict(template_data, card, templates, templates_jsons)

    return templates, templates_jsons


def get_templates_from_list(template_data, card):
    templates, templates_jsons = set(), {}
    _handle_templates_list(template_data, card, templates, templates_jsons)
    return templates, templates_jsons


def _handle_templates_dict(items, card, templates, templates_jsons):
    for template_name, template in items.items():
        if isinstance(template, str):
            register_existing_template(template, templates, templates_jsons)
        elif isinstance(template, dict):
            register_inline_template(
                template_name, template, card, templates, templates_jsons
            )


def _handle_templates_list(items, card, templates, templates_jsons):
    for i, template in enumerate(items):
        if isinstance(template, str):
            register_existing_template(template, templates, templates_jsons)
        elif isinstance(template, dict):
            template_name = f"default_{i}"
            register_inline_template(
                template_name, template, card, templates, templates_jsons
            )


def _handle_plain_dict(data, card, templates, templates_jsons):
    for template_name, template in data.items():
        if isinstance(template, str):
            register_existing_template(template, templates, templates_jsons)
        else:
            register_inline_template(
                template_name, template, card, templates, templates_jsons
            )


def register_existing_template(template, templates, templates_jsons):
    templates.add(template)
    templates_jsons[template] = safe_load_json(get_file_from_item_name(template))


def register_inline_template(template_name, template, card, templates, templates_jsons):
    template_name_prefix = card.replace("cards.", "templates.tmp.") + ".{template}"
    template_name = template_name_prefix.format(template=template_name)
    templates.add(template_name)
    templates_jsons[template_name] = template
    save_temporary_template(template_name, template)


def save_temporary_template(template_name, template_json):
    full_path = os.path.join(CATALOG_DIR, template_name).replace(".", "/") + ".json"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        json.dump(template_json, f, indent=2)


def create_temporary_dict():
    base_temp_path = os.path.join(CATALOG_DIR, "templates")
    temporary_dir_path = os.path.join(base_temp_path, "tmp")

    if os.path.exists(temporary_dir_path):
        shutil.rmtree(temporary_dir_path, ignore_errors=True)

    os.makedirs(temporary_dir_path, exist_ok=True)

    readme_path = os.path.join(temporary_dir_path, "README.txt")
    with open(readme_path, "w") as f:
        f.write(
            "This is a temporary directory used by the UI to store generated templates.\n"
            "It is automatically deleted at the start or end of each run.\n"
            "Safe to ignore or delete manually if needed.\n"
        )

    def cleanup(*args):
        if os.path.exists(temporary_dir_path):
            shutil.rmtree(temporary_dir_path, ignore_errors=True)

    atexit.register(cleanup)


def load_cards_data():
    def is_valid_data(data):
        if (
            "task" not in data
            or isinstance(data["task"], dict)
            or "templates" not in data
        ):
            return False
        return True

    cards_data = {}
    json_data = {}
    unitxt_card_jsons = get_catalog_items_from_dir("cards", UNITXT_DIR)
    private_card_jsons = {}
    if PRIVATE_DIR:
        private_card_jsons = get_catalog_items_from_dir("cards", PRIVATE_DIR)
    card_jsons = {**unitxt_card_jsons, **private_card_jsons}
    cards = card_jsons.keys()
    json_data.update(card_jsons)

    create_temporary_dict()
    for card in cards:
        data = card_jsons[card]
        if not is_valid_data(data):
            continue
        task = data["task"].split("[")[0]
        if task not in cards_data:
            is_augmentable = check_augmentable(task)
        else:
            is_augmentable = cards_data[task][AUGMENTABLE_STR]

        templates, templates_jsons = get_templates(data["templates"], card)
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
    private_items_to_jsons = {}
    unitxt_items_to_jsons = get_catalog_items_from_dir(items_type, UNITXT_DIR)
    if PRIVATE_DIR:
        try:
            private_items_to_jsons = get_catalog_items_from_dir(items_type, PRIVATE_DIR)
        except Exception as e:
            logger.warning(f"Failed to get {items_type}: {e}")

    items_to_jsons = {**unitxt_items_to_jsons, **private_items_to_jsons}
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
