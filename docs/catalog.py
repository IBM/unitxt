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
            yield entry
            yield from custom_walk(entry.path)
        else:
            yield entry


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


class CatalogEntry:
    """A single catalog entry is an artifact json file, or a catalog directory."""

    def __init__(
        self, path: str, is_dir: bool, start_directory: str, relative_path=None
    ):
        self.path = path
        self.is_dir = is_dir
        self.rel_path = (
            relative_path if relative_path else path.replace(start_directory + "/", "")
        )

    @staticmethod
    def from_dir_entry(dir_entry: os.DirEntry, start_directory):
        return CatalogEntry(
            path=dir_entry.path,
            is_dir=dir_entry.is_dir(),
            start_directory=start_directory,
        )

    def is_json(self):
        return not self.is_dir and ".json" in self.rel_path

    def get_label(self):
        label = self.rel_path.replace(".json", "")
        return label.replace("/", ".")

    def get_title(self):
        return Path(self.rel_path).stem

    def get_artifact_doc_path(self, destination_directory):
        return os.path.join(
            destination_directory,
            os.path.dirname(self.rel_path),
            Path(self.rel_path).stem + ".rst",
        )

    def write_dir_contents_to_rst(self, destination_directory, start_directory):
        title = self.get_title()
        if title:
            title = f"{title[0].upper()}{title[1:]}"
        label = self.get_label()
        dir_doc_content = write_title(title, label)

        for sub_dir_entry in os.scandir(self.path):
            sub_catalog_entry = CatalogEntry.from_dir_entry(
                dir_entry=sub_dir_entry, start_directory=start_directory
            )
            sub_label = sub_catalog_entry.get_label()
            sub_name = os.path.basename(sub_dir_entry.path)
            dir_doc_content += f":ref:`{sub_name} <{sub_label}>`\n" f"*************\n\n"

        dir_doc_path = os.path.join(
            destination_directory,
            self.rel_path + ".rst",
        )
        with open(dir_doc_path, "w+") as f:
            f.write(dir_doc_content)

    def write_json_contents_to_rst(self, all_labels, destination_directory):
        artifact = load_json(self.path)
        label = self.get_label()
        content = make_content(artifact, label, all_labels)
        section = write_section(self.get_title(), content, label)
        artifact_doc_path = self.get_artifact_doc_path(
            destination_directory=destination_directory
        )
        Path(artifact_doc_path).parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_doc_path, "w+") as f:
            f.write(section)


class CatalogDocsBuilder:
    """Creates the catalog documentation.

    The builder goes over the catalog, and produces RST doc files for its contents.
    """

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self.start_directory = os.path.join(
            current_dir, "..", "src", "unitxt", "catalog"
        )

    def create_catalog_entries(self):
        return [
            CatalogEntry.from_dir_entry(dir_entry, self.start_directory)
            for dir_entry in custom_walk(self.start_directory)
        ]

    def run(self):
        catalog_entries = self.create_catalog_entries()
        all_labels = {
            catalog_entry.get_label()
            for catalog_entry in catalog_entries
            if catalog_entry.is_json()
        }

        current_directory = os.path.dirname(os.path.abspath(__file__))
        for catalog_entry in catalog_entries:
            if catalog_entry.is_dir:
                catalog_entry.write_dir_contents_to_rst(
                    destination_directory=current_directory,
                    start_directory=self.start_directory,
                )
            elif catalog_entry.is_json():
                catalog_entry.write_json_contents_to_rst(
                    all_labels, destination_directory=current_directory
                )

        catalog_main_entry = CatalogEntry(
            path=self.start_directory + "/",
            is_dir=True,
            start_directory=self.start_directory,
            relative_path="catalog",
        )
        catalog_main_entry.write_dir_contents_to_rst(
            destination_directory=current_directory,
            start_directory=self.start_directory,
        )


if __name__ == "__main__":
    CatalogDocsBuilder().run()
