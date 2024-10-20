import base64
import io
import re
from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from .dict_utils import dict_get
from .operators import FieldOperator, InstanceFieldOperator, PackageRequirementsMixin


class PillowMixin(PackageRequirementsMixin):
    _requirements_list = {"PIL": "pip install pillow"}

    def prepare(self):
        super().prepare()
        import PIL
        from PIL import Image

        self.pil = PIL
        self.image = Image


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
