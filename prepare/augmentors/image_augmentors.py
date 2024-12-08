from unitxt import add_to_catalog
from unitxt.image_operators import GrayScale, GridLines, Oldify, PixelNoise

operator = GrayScale()

add_to_catalog(operator, "augmentors.image.grey_scale", overwrite=True)

operator = GridLines()

add_to_catalog(operator, "augmentors.image.grid_lines", overwrite=True)

operator = PixelNoise()

add_to_catalog(operator, "augmentors.image.white_noise", overwrite=True)

operator = Oldify()

add_to_catalog(operator, "augmentors.image.oldify", overwrite=True)

operator = ImagesAugmentor(operator=ToRGB())

add_to_catalog(operator, "augmentors.image.to_rgb", overwrite=True)
