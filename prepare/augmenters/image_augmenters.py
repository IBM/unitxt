from unitxt import add_to_catalog
from unitxt.image_operators import GrayScale, GridLines, Oldify, PixelNoise

operator = GrayScale()

add_to_catalog(operator, "augmenters.image.grey_scale", overwrite=True)

operator = GridLines()

add_to_catalog(operator, "augmenters.image.grid_lines", overwrite=True)

operator = PixelNoise()

add_to_catalog(operator, "augmenters.image.white_noise", overwrite=True)

operator = Oldify()

add_to_catalog(operator, "augmenters.image.oldify", overwrite=True)
