from src.unitxt import add_to_catalog
from src.unitxt.operators import NullAugmentor

operator = NullAugmentor()

add_to_catalog(operator, "augmentors.no_augmentation", overwrite=True)
