import json
import os
from pathlib import Path

from unitxt.artifact import Artifact
from unitxt.utils import load_json


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


def all_subtypes_of_artifact(artifact):
    if (
        artifact is None
        or isinstance(artifact, str)
        or isinstance(artifact, bool)
        or isinstance(artifact, int)
        or isinstance(artifact, float)
    ):
        return []
    if isinstance(artifact, list):
        to_return = []
        for art in artifact:
            to_return.extend(all_subtypes_of_artifact(art))
        return to_return
    # artifact is a dict
    to_return = []
    for key, value in artifact.items():
        if isinstance(value, str):
            if key == "type":
                to_return.append(value)
        else:
            to_return.extend(all_subtypes_of_artifact(value))
    return to_return


def make_content(artifact, label, all_labels):
    artifact_type = artifact["type"]
    artifact_class = Artifact._class_register.get(artifact_type)
    type_class_name = artifact_class.__name__
    artifact_class_id = f"{artifact_class.__module__}.{type_class_name}"
    result = (
        f".. note:: ID: ``{label}``  |  Type: :class:`{type_class_name} <{artifact_class_id}>`\n\n"
        f"   .. code-block:: json\n\n      "
    )
    result += (
        json.dumps(artifact, sort_keys=True, indent=4, ensure_ascii=False).replace(
            "\n", "\n      "
        )
        + "\n"
    )

    if artifact_class.__doc__:
        explanation_str = f"Explanation about `{type_class_name}`"
        result += f"\n{explanation_str}\n"
        result += "+" * len(explanation_str) + "\n\n"
        result += artifact_class.__doc__ + "\n"

    subtypes = all_subtypes_of_artifact(artifact)
    subtypes = list(set(subtypes))
    subtypes.remove(artifact_type)  # this was already documented
    for subtype in subtypes:
        subtype_class = Artifact._class_register.get(subtype)
        subtype_class_name = subtype_class.__name__
        if subtype_class.__doc__:
            explanation_str = f"Explanation about `{subtype_class_name}`"
            result += f"\n{explanation_str}\n"
            result += "+" * len(explanation_str) + "\n\n"
            result += subtype_class.__doc__ + "\n"

    references = []
    for label in all_labels:
        label_no_catalog = label[8:]  # skip over the prefix 'catalog.'
        if f'"{label_no_catalog}"' in result:
            references.append(f":ref:`{label_no_catalog} <{label}>`")
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
        return not self.is_dir and self.rel_path.endswith(".json")

    def get_label(self):
        label = self.rel_path.replace(".json", "")
        label = label.replace("/", ".")
        if not self.is_main_catalog_entry():
            label = "catalog." + label
        return label

    def is_main_catalog_entry(self):
        return self.rel_path.startswith("catalog")

    def get_title(self):
        return Path(self.rel_path).stem

    def get_artifact_doc_path(self, destination_directory, with_extension=True):
        """Return the path to an RST file which describes this object."""
        dirname = os.path.dirname(self.rel_path)
        if dirname:
            dirname = dirname.replace(os.path.sep, ".") + "."
        if not self.is_main_catalog_entry():
            # add prefix to the rst file path, if not the main catalog file
            dirname = "catalog." + dirname
        result = os.path.join(
            destination_directory,
            dirname + Path(self.rel_path).stem,
        )

        if with_extension:
            result += ".rst"
        return result

    def write_dir_contents_to_rst(self, destination_directory, start_directory):
        title = self.get_title()
        if title:
            title = f"{title[0].upper()}{title[1:]}"
        label = self.get_label()
        dir_doc_content = write_title(title, label)
        dir_doc_content += ".. toctree::\n   :maxdepth: 1\n\n"

        sub_catalog_entries = [
            CatalogEntry.from_dir_entry(
                dir_entry=sub_dir_entry, start_directory=start_directory
            )
            for sub_dir_entry in os.scandir(self.path)
        ]

        sub_dir_entries = [entry for entry in sub_catalog_entries if entry.is_dir]
        sub_dir_entries.sort(key=lambda entry: entry.path)
        for sub_dir_entry in sub_dir_entries:
            sub_name = sub_dir_entry.get_artifact_doc_path(
                destination_directory="", with_extension=False
            )
            dir_doc_content += f"   {sub_name}\n"

        sub_file_entries = [entry for entry in sub_catalog_entries if not entry.is_dir]
        sub_file_entries.sort(key=lambda entry: entry.path)
        for sub_file_entry in sub_file_entries:
            sub_name = sub_file_entry.get_artifact_doc_path(
                destination_directory="", with_extension=False
            )
            dir_doc_content += f"   {sub_name}\n"

        dir_doc_path = self.get_artifact_doc_path(destination_directory)
        Path(dir_doc_path).parent.mkdir(exist_ok=True, parents=True)
        with open(dir_doc_path, "w+") as f:
            f.write(dir_doc_content)

    def write_json_contents_to_rst(self, all_labels, destination_directory):
        artifact = load_json(self.path)
        label = self.get_label()
        content = make_content(artifact, label, all_labels)

        underline_char = "-"
        underline = underline_char * len(self.get_title())
        artifact_doc_contents = (
            f".. _{label}:\n\n"
            f"----------\n\n"
            f"{self.get_title()}\n"
            f"{underline}\n\n"
            f"{content}\n\n"
            f"|\n"
            f"|\n\n"
        )
        artifact_doc_path = self.get_artifact_doc_path(
            destination_directory=destination_directory
        )
        Path(artifact_doc_path).parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_doc_path, "w+") as f:
            f.write(artifact_doc_contents)


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


def replace_in_file(source_str, target_str, file_path):
    """Replace all occurrences of source_str with target_str in the file specified by file_path.

    Parameters:
    - source_str: The string to be replaced.
    - target_str: The string to replace with.
    - file_path: The path to the file where the replacement should occur.
    """
    with open(file_path) as file:
        file_contents = file.read()
    modified_contents = file_contents.replace(source_str, target_str)
    with open(file_path, "w") as file:
        file.write(modified_contents)


def create_catalog_docs():
    CatalogDocsBuilder().run()
    current_dir = os.path.dirname(__file__)
    catalog_rst_file = os.path.join(current_dir, "catalog.rst")
    source_str = "Catalog\n-------\n"
    target_str = (
        "Catalog\n-------\n\n"
        "The catalog is a place for the Unitxt community to share assets used for processing datasets.\n\n"
        "You can load directly from the catalog with `fetch_artifact('artifact.id')`.\n"
        "You can also place anywhere in Unitxt catalog ids instead of realclasses\n"
        "Finally you can also state an ID with a modification to the asset parameters `asset.id[key=value]`.\n"
    )
    replace_in_file(source_str, target_str, catalog_rst_file)


if __name__ == "__main__":
    create_catalog_docs()
