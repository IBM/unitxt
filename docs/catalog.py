import json
import os
from pathlib import Path

from unitxt.artifact import Artifact
from unitxt.utils import load_json


def write_section(title, content, label):
    underline_char = "-"
    underline = underline_char * len(title)
    return (
        f".. _{label}:\n\n----------\n\n{title}\n{underline}\n\n{content}\n\n|\n|\n\n"
    )


def write_title(title, label):
    underline_char = "-"
    underline = underline_char * len(title)

    return f".. _{label}:\n\n----------\n\n{title}\n{underline}\n\n"


def custom_walk(top):
    for entry in os.scandir(top):
        if entry.is_dir():
            yield (entry.path, True)
            yield from custom_walk(entry.path)
        else:
            yield (entry.path, False)


def make_content(artifact, label, all_labels):
    artifact_type = artifact["type"]
    artifact_class = Artifact._class_register.get(artifact_type)
    class_name = artifact_class.__name__
    artifact_class_id = f"{artifact_class.__module__}.{class_name}"
    result = f".. note:: ID: ``{label}``  |  Type: :class:`{class_name} <{artifact_class_id}>`\n\n   .. code-block:: json\n\n      "
    result += (
        json.dumps(artifact, sort_keys=True, indent=4, ensure_ascii=False).replace(
            "\n", "\n      "
        )
        + "\n"
    )

    if artifact_class.__doc__:
        result += artifact_class.__doc__ + "\n"

    references = []
    for label in all_labels:
        if f'"{label}"' in result:
            references.append(f":ref:`{label} <{label}>`")
    if len(references) > 0:
        result += "\nReferences: " + ", ".join(references)
    return result


def build_catalog_rst():
    all_labels = set()
    current_dir = os.path.dirname(__file__)
    start_directory = os.path.join(current_dir, "..", "src", "unitxt", "catalog")

    for path, is_dir in custom_walk(start_directory):
        rel_path = path.replace(start_directory + "/", "")
        if not is_dir and ".json" in rel_path:
            rel_path = rel_path.replace(".json", "")
            label = rel_path.replace("/", ".")
            all_labels.add(label)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    for path, is_dir in custom_walk(start_directory):
        rel_path = path.replace(start_directory + "/", "")
        if is_dir:
            create_dir_contents_rst(start_directory, current_directory, path, rel_path)
        else:
            if ".json" in rel_path:
                artifact = load_json(path)
                label = rel_path.replace("/", ".")
                content = make_content(artifact, label, all_labels)
                section = write_section(rel_path.split("/")[-1], content, label)

                artifact_doc_path = os.path.join(
                    current_directory,
                    os.path.dirname(rel_path),
                    Path(rel_path).stem + ".rst",
                )
                Path(artifact_doc_path).parent.mkdir(parents=True, exist_ok=True)
                with open(artifact_doc_path, "w+") as f:
                    f.write(section)

    create_dir_contents_rst(
        start_directory,
        current_directory,
        dir_path=start_directory,
        rel_path="catalog",
    )


def create_dir_contents_rst(start_directory, current_directory, dir_path, rel_path):
    dir_doc_path = os.path.join(
        current_directory,
        rel_path + ".rst",
    )
    title = rel_path.split("/")[-1]
    if title:
        title = f"{title[0].upper()}{title[1:]}"
    label = rel_path.replace("/", ".")
    dir_doc_content = write_title(title, label)

    for dir_entry in os.scandir(dir_path):
        sub_rel_path = dir_entry.path.replace(start_directory + "/", "")
        sub_label = sub_rel_path.replace("/", ".")
        sub_name = os.path.basename(dir_entry.path)
        dir_doc_content += f":ref:`{sub_name} <{sub_label}>`\n\n"
    with open(dir_doc_path, "w+") as f:
        f.write(dir_doc_content)


if __name__ == "__main__":
    build_catalog_rst()
