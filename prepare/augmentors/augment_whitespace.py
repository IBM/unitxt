from src.unitxt import add_to_catalog
from src.unitxt.operators import AugmentWhitespace,Augmentor

operator = AugmentWhitespace()

add_to_catalog(operator, "augmentors.augment_whitespace", overwrite=True)

operator = Augmentor()

add_to_catalog(operator, "augmentors.no_augmentation", overwrite=True)
