from unitxt.artifact import ArtifactLink
from unitxt.augmentors import (
    AugmentPrefixSuffix,
    AugmentWhitespace,
)
from unitxt.catalog import add_to_catalog

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
add_to_catalog(
    ArtifactLink("augmentors.text.whitespace_prefix_suffix"),
    "augmentors.augment_whitespace_prefix_and_suffix_task_input",
    overwrite=True,
)
add_to_catalog(AugmentWhitespace(), "augmentors.text.white_space", overwrite=True)

add_to_catalog(
    ArtifactLink("augmentors.text.whitespace_prefix_suffix"),
    "augmentors.augment_whitespace_task_input",
    overwrite=True,
)
