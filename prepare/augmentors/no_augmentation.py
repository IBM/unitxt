from unitxt import add_to_catalog
from unitxt.operators import NullAugmentor

operator = NullAugmentor()

add_to_catalog(operator, "augmentors.no_augmentation", overwrite=True)
