from unitxt import add_to_catalog
from unitxt.augmentors import (
    ImagesAugmentor,
)
from unitxt.image_operators import GrayScale, GridLines, Oldify, PixelNoise, ToRGB

operator = ImagesAugmentor(operator=GrayScale())

add_to_catalog(operator, "augmentors.image.grey_scale", overwrite=True)

operator = ImagesAugmentor(operator=GridLines())

add_to_catalog(operator, "augmentors.image.grid_lines", overwrite=True)

operator = ImagesAugmentor(operator=PixelNoise())

add_to_catalog(operator, "augmentors.image.white_noise", overwrite=True)

operator = ImagesAugmentor(operator=Oldify())

add_to_catalog(operator, "augmentors.image.oldify", overwrite=True)

operator = ImagesAugmentor(operator=ToRGB())

add_to_catalog(operator, "augmentors.image.to_rgb", overwrite=True)
