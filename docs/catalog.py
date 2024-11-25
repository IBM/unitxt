import os
import re
from functools import lru_cache
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import YamlLexer
from unitxt.artifact import Artifact
from unitxt.text_utils import print_dict_as_yaml
from unitxt.utils import load_json


def dict_to_syntax_highlighted_html(nested_dict):
    # Convert the dictionary to a YAML string with indentation
    yaml_str = print_dict_as_yaml(nested_dict)
    # Initialize the HTML formatter with no additional wrapper
    formatter = HtmlFormatter(nowrap=True)
    # Apply syntax highlighting
    return highlight(yaml_str, YamlLexer(), formatter)


def write_title(title, label):
    title = f"📁 {title}"
    wrap_char = "="
    wrap = wrap_char * (len(title) + 1)

    return f".. _{label}:\n\n{wrap}\n{title}\n{wrap}\n\n"


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
            if key == "__type__":
                to_return.append(value)
        else:
            to_return.extend(all_subtypes_of_artifact(value))
    return to_return


def get_all_type_elements(nested_dict):
    type_elements = set()

    def recursive_search(d):
        if isinstance(d, dict):
            d.pop("__description__", None)
            d.pop("__tags__", None)
            for key, value in d.items():
                if key == "__type__":
                    type_elements.add(value)
                elif isinstance(value, dict):
                    recursive_search(value)
                elif isinstance(value, list):
                    for item in value:
                        recursive_search(item)

    recursive_search(nested_dict)
    return list(type_elements)


@lru_cache(maxsize=None)
def artifact_type_to_link(artifact_type):
    artifact_class = Artifact._class_register.get(artifact_type)
    type_class_name = artifact_class.__name__
    artifact_class_id = f"{artifact_class.__module__}.{type_class_name}"
    return f'<a class="reference internal" href="../{artifact_class.__module__}.html#{artifact_class_id}" title="{artifact_class_id}"><code class="xref py py-class docutils literal notranslate"><span class="pre">{type_class_name}</span></code></a>'


# flake8: noqa: C901
def make_content(artifact, label, all_labels):
    artifact_type = artifact["__type__"]
    artifact_class = Artifact._class_register.get(artifact_type)
    type_class_name = artifact_class.__name__
    catalog_id = label.replace("catalog.", "")

    result = ""

    if "__description__" in artifact and artifact["__description__"] is not None:
        result += "\n" + artifact["__description__"] + "\n"
        result += "\n"
        artifact.pop("__description__")  # to not show again in the yaml

    if "__tags__" in artifact and artifact["__tags__"] is not None:
        result += "\nTags: "
        tags = []
        for k, v in artifact["__tags__"].items():
            tags.append(f"``{k}:{v!s}``")
        result += ",  ".join(tags) + "\n\n"
        artifact.pop("__tags__")  # to not show again in the yaml

    result += ".. raw:: html\n\n   "

    type_elements = get_all_type_elements(artifact)

    html_for_dict = dict_to_syntax_highlighted_html(artifact)

    pairs = []
    references = []
    for i, label in enumerate(all_labels):
        label_no_catalog = label[8:]  # skip over the prefix 'catalog.'
        if label_no_catalog in html_for_dict:
            html_for_dict = html_for_dict.replace(
                label_no_catalog,
                f"[[[[{i}]]]]",
            )
            pairs.append((f"[[[[{i}]]]]", label))
            references.append(f":ref:`{label_no_catalog} <{label}>`")

    for key, label in pairs:
        label_no_catalog = label[8:]  # skip over the prefix 'catalog.'
        label_replace_dot_by_hyphen = label.replace(".", "-")
        html_for_dict = html_for_dict.replace(
            key,
            f'<a class="reference internal" href="{label}.html#{label_replace_dot_by_hyphen}"><span class="std std-ref">{label_no_catalog}</span></a>',
        )

    for type_name in type_elements:
        source = f'<span class="nt">__type__</span><span class="p">:</span><span class="w"> </span><span class="l l-Scalar l-Scalar-Plain">{type_name}</span>'
        target = artifact_type_to_link(type_name)
        html_for_dict = html_for_dict.replace(
            source,
            '<span class="nt">&quot;type&quot;</span><span class="p">:</span><span class="w"> </span>'
            + target,
        )

    pattern = r'(<span class="nt">)&quot;(.*?)&quot;(</span>)'

    # Replacement function
    html_for_dict = re.sub(pattern, r"\1\2\3", html_for_dict)
    source_link = f"""<a class="reference external" href="https://github.com/IBM/unitxt/blob/main/src/unitxt/catalog/{catalog_id.replace('.','/')}.json"><span class="viewcode-link"><span class="pre">[source]</span></span></a>"""
    html_for_dict = f"""<div class="admonition note">
<p class="admonition-title">{catalog_id}</p>
<div class="highlight-json notranslate">
<div class="highlight"><pre>
{html_for_dict.strip()}
</pre>{source_link}</div></div>
</div>""".replace("\n", "\n    ")

    result += "    " + html_for_dict + "\n"

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

    if len(references) > 0:
        result += "\nReferences: " + ", ".join(references)
    return (
        result
        + "\n\n\n\n\nRead more about catalog usage :ref:`here <using_catalog>`.\n\n"
    )


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
        label = label.replace(os.path.sep, ".")
        if not self.is_main_catalog_entry():
            label = "catalog." + label
        if self.is_dir:
            label = "dir_" + label
        return label

    def is_main_catalog_entry(self):
        return self.rel_path.startswith("catalog")

    def get_title(self):
        title = Path(self.rel_path).stem
        return title.replace("_", " ").title()

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
            "catalog",
            dirname + Path(self.rel_path).stem,
        )
        if self.is_dir:
            result += ".__dir__"

        if with_extension:
            result += ".rst"

        return result

    def write_dir_contents_to_rst(self, destination_directory, start_directory):
        title = self.get_title()
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
            if sub_name.startswith("catalog/"):
                sub_name = sub_name.replace("catalog/", "", 1)
            dir_doc_content += f"   {sub_name}\n"

        sub_file_entries = [entry for entry in sub_catalog_entries if not entry.is_dir]
        sub_file_entries.sort(key=lambda entry: entry.path)
        for sub_file_entry in sub_file_entries:
            sub_name = sub_file_entry.get_artifact_doc_path(
                destination_directory="", with_extension=False
            )
            if sub_name.startswith("catalog/"):
                sub_name = sub_name.replace("catalog/", "", 1)
            dir_doc_content += f"   {sub_name}\n"

        dir_doc_path = self.get_artifact_doc_path(destination_directory)
        Path(dir_doc_path).parent.mkdir(exist_ok=True, parents=True)
        with open(dir_doc_path, "w+") as f:
            f.write(
                dir_doc_content
                + "\n\n\n\n\nRead more about catalog usage :ref:`here <using_catalog>`.\n\n"
            )

    def write_json_contents_to_rst(self, all_labels, destination_directory):
        artifact = load_json(self.path)
        label = self.get_label()
        deprecated_in_title = ""
        deprecated_message = ""
        role_red = ""
        if (
            "__deprecated_msg__" in artifact
            and artifact["__deprecated_msg__"] is not None
        ):
            deprecated_in_title = " :red:`[deprecated]`"
            deprecated_message = (
                "**Deprecation message:** " + artifact["__deprecated_msg__"] + "\n\n"
            )
            role_red = ".. role:: red\n\n"
            artifact.pop("__deprecated_msg__")

        content = make_content(artifact, label, all_labels)
        title_char = "="
        title = "📄 " + self.get_title() + deprecated_in_title
        title_wrapper = title_char * (len(title) + 1)
        artifact_doc_contents = (
            f"{role_red}"
            f".. _{label}:\n\n"
            f"{title_wrapper}\n"
            f"{title}\n"
            f"{title_wrapper}\n\n"
            f"{deprecated_message}"
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

        all_labels = sorted(all_labels, key=len, reverse=True)

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


if __name__ == "__main__":
    create_catalog_docs()
