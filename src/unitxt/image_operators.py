import base64
import io
import re
from abc import abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from datasets import Image as DatasetsImage

from .augmentors import TaskInputsAugmentor
from .dict_utils import dict_get
from .operator import PackageRequirementsMixin
from .operators import FieldOperator, InstanceFieldOperator
from .settings_utils import get_constants
from .type_utils import isoftype
from .types import Image

constants = get_constants()

datasets_image = DatasetsImage()


def _image_to_bytes(image, format="JPEG"):
    import base64

    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class ImageDataString(str):
    def __repr__(self) -> str:
        if len(self) > 30:
            return '<ImageDataString "' + self[:30] + '...">'
        return super().__repr__()


def image_to_data_url(image: Image, default_format="JPEG"):
    """Convert an image to a data URL.

    https://developer.mozilla.org/en-US/docs/Web/URI/Schemes/data
    """
    image_format = image["format"] if image["format"] else default_format
    base64_image = _image_to_bytes(image["image"], format=image_format.upper())
    return ImageDataString(f"data:image/{image_format.lower()};base64,{base64_image}")


def _bytes_to_image(b64_string):
    import base64
    import io

    from PIL import Image

    # Decode the base64-encoded string
    decoded_bytes = base64.b64decode(b64_string)
    # Open the image from the decoded bytes
    return Image.open(io.BytesIO(decoded_bytes))


def data_url_to_image(data_url: str):
    import re

    # Verify that the string is a data URL
    if not data_url.startswith("data:"):
        raise ValueError("Invalid data URL")

    # Extract the base64 data using a regular expression
    match = re.match(r"data:image/(.*?);base64,(.*)", data_url)
    if not match:
        raise ValueError("Invalid data URL format")

    # Extract image format and base64 data
    image_format, b64_data = match.groups()

    # Use _bytes_to_image to convert base64 data to an image
    return _bytes_to_image(b64_data)


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


def extract_images(instance):
    regex = r"<" + f"{constants.image_tag}" + r'\s+src=["\'](.*?)["\']'
    image_sources = re.findall(regex, instance["source"])
    images = []
    for image_source in image_sources:
        image = dict_get(instance, image_source)
        images.append(image)
    return images


class EncodeImageToString(FieldOperator):
    image_format: str = "JPEG"

    def encode_image_to_base64(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format=self.image_format)
        return ImageDataString(base64.b64encode(buffer.getvalue()).decode("utf-8"))

    def process_value(self, value: Any) -> Any:
        return {"image": self.encode_image_to_base64(value)}


class DecodeImage(FieldOperator, PillowMixin):
    def process_value(self, value: str) -> Any:
        image_data = base64.b64decode(value)
        return self.image.open(io.BytesIO(image_data))


class ToImage(InstanceFieldOperator):
    def process_instance_value(self, value: Any, instance: Dict[str, Any]) -> Image:
        return {
            "image": value,
            "format": value.format if value.format is not None else "JPEG",
        }


class ImageFieldOperator(FieldOperator, PillowMixin):
    @abstractmethod
    def process_image(self, image: Any):
        pass

    def process_value(self, value: Image) -> Any:
        if not isinstance(value["image"], self.image.Image):
            raise ValueError(f"ImageFieldOperator requires image, got {type(value)}.")
        value["image"] = self.process_image(value["image"])
        return value


class ImageAugmentor(TaskInputsAugmentor, PillowMixin):
    augmented_type: object = Image

    @abstractmethod
    def process_image(self, image: Any):
        pass

    def process_value(self, value: Image) -> Any:
        if not isoftype(value, Image):
            return value
        value["image"] = self.process_image(value["image"])
        return value


class GrayScale(ImageAugmentor):
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


class GridLines(ImageAugmentor):
    """A class that overlays a fixed number of evenly spaced horizontal and vertical lines on an image.

    Args:
        num_lines (int):
            The number of horizontal and vertical lines to add.
        line_thickness (int):
            Thickness of each line in pixels.
        line_color (Tuple[int, int, int]):
            RGB color of the grid lines.

    Methods:
        process_image(image): Adds grid lines to the provided image and returns the modified image.
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


class PixelNoise(ImageAugmentor):
    """A class that overlays a mask of randomly colored nxn squares across an image based on a specified noise rate.

    Args:
        square_size (int):
            Size of each square in pixels.
        noise_rate (float):
            Proportion of the image that should be affected by noise (0 to 1).

    Methods:
        process_image(image):
            Adds the random square mask to the provided image and returns the modified image.
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


class Oldify(ImageAugmentor):
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
