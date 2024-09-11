from unitxt import add_to_catalog
from unitxt.augmentors import (
    ImagesAugmentor,
)
from unitxt.image_operators import GrayScale

operator = ImagesAugmentor(operator=GrayScale())

add_to_catalog(operator, "augmentors.image.grey_scale", overwrite=True)
