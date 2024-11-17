from unitxt import add_to_catalog
from unitxt.augmenters import NullAugmenter

operator = NullAugmenter()

add_to_catalog(operator, "augmenters.no_augmentation", overwrite=True)
