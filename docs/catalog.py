import json
import os

import unitxt
from unitxt.artifact import Artifact

depth_levels = ["=", "-", "^", '"', "'", "~", "*", "+", "#", "_"]


def write_section(title, content, label, depth):
    if depth < 0 or depth > len(depth_levels) - 1:
        raise ValueError(
            f"Depth should be between 0 and {len(depth_levels)}, Got {depth}"
        )

    underline_char = depth_levels[depth]
    underline = underline_char * len(title)
    return (
        f".. _{label}:\n\n----------\n\n{title}\n{underline}\n\n{content}\n\n|\n|\n\n"
    )


def write_title(title, label, depth):
    if depth < 0 or depth > len(depth_levels) - 1:
        raise ValueError(
            f"Depth should be between 0 and {len(depth_levels)}, Got {depth}"
        )

    underline_char = depth_levels[depth]
    underline = underline_char * len(title)

    return f".. _{label}:\n\n----------\n\n{title}\n{underline}\n\n"


def custom_walk(top, depth=0):
    for entry in os.scandir(top):
        if entry.is_dir():
            yield (entry.path, True, depth)
            yield from custom_walk(entry.path, depth + 1)
        else:
            yield (entry.path, False, depth)


def make_content(artifact, label, all_labels={}):
    artifact_type = artifact["type"]
    artifact_class = Artifact._class_register.get(artifact_type)
    class_name = artifact_class.__name__
    artifact_class_id = f"{artifact_class.__module__}.{class_name}"
    result = f".. note:: ID: ``{label}``  |  Type: :class:`{class_name} <{artifact_class_id}>`\n\n   .. code-block:: json\n\n      "
    result += (
        json.dumps(artifact, sort_keys=True, indent=4).replace("\n", "\n      ") + "\n"
    )

    references = []
    for l in all_labels:
        if f'"{l}"' in result:
            references.append(f":ref:`{l} <{l}>`")
    if len(references) > 0:
        result += "\nReferences: " + ", ".join(references)
    return result


prints = [write_title("Catalog", "catalog", 0)]
all_labels = set()
start_directory = unitxt.local_catalog_path

for path, is_dir, depth in custom_walk(start_directory):
    rel_path = path.replace(start_directory, "")
    if not is_dir and ".json" in rel_path:
        rel_path = rel_path.replace(".json", "")
        label = rel_path.replace("/", ".")[1:]
        all_labels.add(label)

for path, is_dir, depth in custom_walk(start_directory):
    rel_path = path.replace(start_directory, "")
    if is_dir:
        prints.append(
            write_title(rel_path.split("/")[-1], rel_path.replace("/", "."), depth + 1)
        )
    else:
        if ".json" in rel_path:
            rel_path = rel_path.replace(".json", "")
            with open(path, "r") as f:
                artifact = json.load(f)
            label = rel_path.replace("/", ".")[1:]
            content = make_content(artifact, label, all_labels)
            section = write_section(rel_path.split("/")[-1], content, label, depth + 1)
            prints.append(section)

current_directory = os.path.dirname(os.path.abspath(__file__))
target_file = os.path.join(current_directory, "catalog.rst")

with open(target_file, "w+") as f:
    f.write("\n\n".join(prints))
