from unitxt import add_to_catalog
from unitxt.splitters import SplitRandomMix

add_to_catalog(
    SplitRandomMix(
        {"train": "train[95%]", "validation": "train[5%]", "test": "validation"}
    ),
    "splitters.small_no_test",
    overwrite=True,
)
add_to_catalog(
    SplitRandomMix(
        {"train": "train[99%]", "validation": "train[1%]", "test": "validation"}
    ),
    "splitters.large_no_test",
    overwrite=True,
)  # Note, this would be change in future version to a constant amount of examples (500) instead of 1% of the examples
add_to_catalog(
    SplitRandomMix({"train": "train[99%]", "validation": "train[1%]", "test": "test"}),
    "splitters.large_no_dev",
    overwrite=True,
)  # Note, this would be change in future version to a constant amount of examples (500) instead of 1% of the examples
add_to_catalog(
    SplitRandomMix({"train": "train[95%]", "validation": "train[5%]", "test": "test"}),
    "splitters.small_no_dev",
    overwrite=True,
)
add_to_catalog(
    SplitRandomMix({"train": "test[0%]", "validation": "test[0%]", "test": "test"}),
    "splitters.test_only",
    overwrite=True,
)
