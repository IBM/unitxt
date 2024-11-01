import base64
import io
import re
from abc import abstractmethod
from typing import Any, Dict, Tuple

import numpy as np

from .dict_utils import dict_get
from .operator import PackageRequirementsMixin
from .operators import FieldOperator, InstanceFieldOperator


class PillowMixin(PackageRequirementsMixin):
    _requirements_list = {"PIL": "pip install pillow"}

    def prepare(self):
        super().prepare()
        import PIL
        from PIL import Image, ImageEnhance, ImageFilter

        self.pil = PIL
        self.image = Image
        self.enhance = ImageEnhance
        self.filter = ImageFilter


def extract_images(text, instance):
    regex = r'<img\s+src=["\'](.*?)["\']'
    image_sources = re.findall(regex, text)
    images = []
    for image_source in image_sources:
        image = dict_get(instance, image_source)
        images.append(image)
    return images


class DecodeImage(FieldOperator, PillowMixin):
    def decode_base64_to_image(self, base64_string):
        image_data = base64.b64decode(base64_string)
        return self.image.open(io.BytesIO(image_data))

    def process_value(self, value: Any) -> Any:
        return {"image": self.decode_base64_to_image(value)}


class ToImage(InstanceFieldOperator):
    def process_instance_value(self, value: Any, instance: Dict[str, Any]):
        return {"image": value}


class ImageFieldOperator(FieldOperator, PillowMixin):
    @abstractmethod
    def process_image(self, image):
        pass

    def process_value(self, value: Any) -> Any:
        if not isinstance(value, self.image.Image):
            raise ValueError(f"ImageFieldOperator requires image, got {type(value)}.")
        return self.process_image(value)


class GrayScale(ImageFieldOperator):
    def process_image(self, image):
        # Convert the image to grayscale
        grayscale_image = image.convert("L")

        # Convert the grayscale image to a NumPy array
        grayscale_array = np.array(grayscale_image)

        # Add a dummy channel dimension to make it (height, width, 1)
        grayscale_array = np.expand_dims(grayscale_array, axis=-1)

        # Repeat the channel to have (height, width, 3) if needed for compatibility
        grayscale_array = np.repeat(grayscale_array, 3, axis=-1)

        # Convert back to a PIL image with 3 channels
        return self.image.fromarray(grayscale_array)


class GridLines(ImageFieldOperator):
    """A class that overlays a fixed number of evenly spaced horizontal and vertical lines on an image.

    Attributes:
    - num_lines (int): The number of horizontal and vertical lines to add.
    - line_thickness (int): Thickness of each line in pixels.
    - line_color (Tuple[int, int, int]): RGB color of the grid lines.

    Methods:
    - process_image(image): Adds grid lines to the provided image and returns the modified image.
    """

    num_lines: int = 128
    line_thickness: int = 1
    line_color: Tuple[int, int, int] = (255, 255, 255)

    def process_image(self, image):
        image_array = np.array(image)

        # Determine image dimensions
        height, width, _ = image_array.shape

        # Calculate spacing for the lines based on image size and number of lines
        horizontal_spacing = height // (self.num_lines + 1)
        vertical_spacing = width // (self.num_lines + 1)

        # Add horizontal lines
        for i in range(1, self.num_lines + 1):
            y = i * horizontal_spacing
            image_array[y : y + self.line_thickness, :, :] = self.line_color

        # Add vertical lines
        for i in range(1, self.num_lines + 1):
            x = i * vertical_spacing
            image_array[:, x : x + self.line_thickness, :] = self.line_color

        # Convert back to a PIL image
        return self.image.fromarray(image_array)


class PixelNoise(ImageFieldOperator):
    """A class that overlays a mask of randomly colored nxn squares across an image based on a specified noise rate.

    Attributes:
    - square_size (int): Size of each square in pixels.
    - noise_rate (float): Proportion of the image that should be affected by noise (0 to 1).

    Methods:
    - process_image(image): Adds the random square mask to the provided image and returns the modified image.
    """

    square_size: int = 1
    noise_rate: float = 0.3  # Percentage of squares to be randomly colored

    def process_image(self, image):
        image_array = np.array(image)
        height, width, channels = image_array.shape

        # Calculate grid dimensions
        y_squares = height // self.square_size
        x_squares = width // self.square_size

        # Create a grid indicating where to apply the mask
        noise_mask = np.random.rand(y_squares, x_squares) < self.noise_rate

        # Generate random colors for each square
        colors = np.random.randint(
            0, 256, (y_squares, x_squares, channels), dtype=np.uint8
        )

        # Expand the mask and colors to the size of the image array
        mask_expanded = np.repeat(
            np.repeat(noise_mask, self.square_size, axis=0), self.square_size, axis=1
        )
        colors_expanded = np.repeat(
            np.repeat(colors, self.square_size, axis=0), self.square_size, axis=1
        )

        # Reshape `mask_expanded` to add the color channel dimension
        mask_expanded = np.repeat(mask_expanded[:, :, np.newaxis], channels, axis=2)

        # Apply colors where the mask is true using element-wise assignment
        image_array = np.where(mask_expanded, colors_expanded, image_array)

        # Convert back to a PIL image
        return self.image.fromarray(image_array)


class Oldify(ImageFieldOperator):
    noise_strength: int = 30
    tint_strength: float = 0.4  # Percentage of squares to be randomly colored

    def process_image(self, image):
        # Convert to a numpy array for manipulation
        image_array = np.array(image)

        # Step 1: Add a slight yellowish tint
        yellow_tint = np.array([255, 228, 170], dtype=np.uint8)  # Aged paper-like color
        tinted_image_array = (
            image_array * (1 - self.tint_strength) + yellow_tint * self.tint_strength
        ).astype(np.uint8)

        # Step 2: Add noise for a "film grain" effect
        noise = np.random.normal(0, self.noise_strength, image_array.shape).astype(
            np.int16
        )
        noisy_image_array = np.clip(tinted_image_array + noise, 0, 255).astype(np.uint8)

        # Step 3: Convert back to a PIL Image for additional processing
        old_image = self.image.fromarray(noisy_image_array)

        # Step 4: Apply a slight blur to mimic an older lens or slight wear
        old_image = old_image.filter(self.filter.GaussianBlur(radius=1))

        # Step 5: Adjust contrast and brightness to give it a "faded" look
        enhancer = self.enhance.Contrast(old_image)
        old_image = enhancer.enhance(0.6)  # Lower contrast

        enhancer = self.enhance.Brightness(old_image)
        return enhancer.enhance(1.2)  # Slightly increased brightness


class ToRGB(ImageFieldOperator):
    def process_image(self, image):
        return image.convert("RGB")
