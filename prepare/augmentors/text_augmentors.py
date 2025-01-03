from unitxt.augmentors import (
    AugmentPrefixSuffix,
    AugmentWhitespace,
)
from unitxt.catalog import add_link_to_catalog, add_to_catalog

pattern_distribution = {
    " ": 20,
    "\\t": 10,
    "\\n": 40,
    "": 30,
}

operator = AugmentPrefixSuffix(
    prefixes=pattern_distribution,
    suffixes=pattern_distribution,
    prefix_len=5,
    suffix_len=5,
    remove_existing_whitespaces=True,
)

add_to_catalog(operator, "augmentors.text.whitespace_prefix_suffix", overwrite=True)
add_link_to_catalog(
    artifact_linked_to="augmentors.text.whitespace_prefix_suffix",
    name="augmentors.augment_whitespace_prefix_and_suffix_task_input",
    deprecate=True,
    overwrite=True,
)

add_to_catalog(AugmentWhitespace(), "augmentors.text.white_space", overwrite=True)

add_link_to_catalog(
    artifact_linked_to="augmentors.text.whitespace_prefix_suffix",
    name="augmentors.augment_whitespace_task_input",
    deprecate=True,
    overwrite=True,
)
