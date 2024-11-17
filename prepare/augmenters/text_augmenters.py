from unitxt.augmenters import (
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

add_to_catalog(operator, "augmenters.text.prefix_suffix", overwrite=True)

add_to_catalog(AugmentWhitespace(), "augmenters.text.white_space", overwrite=True)
