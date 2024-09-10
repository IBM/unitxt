import base64
import io
import re
from typing import Any, Dict

from .dict_utils import dict_get
from .operators import FieldOperator, InstanceFieldOperator


def extract_images(text, instance):
    regex = r'<img\s+src=["\'](.*?)["\']'
    image_sources = re.findall(regex, text)
    images = []
    for image_source in image_sources:
        image = dict_get(instance, image_source)
        images.append(image)
    return images


class DecodeImage(FieldOperator):
    _requirements_list = {"PIL": "Please install pillow with `pip install pillow`."}

    def prepare(self):
        from PIL import Image

        self.image = Image

    def decode_base64_to_image(self, base64_string):
        img_data = base64.b64decode(base64_string)
        return self.image.open(io.BytesIO(img_data))

    def process_value(self, value: Any) -> Any:
        return {"image": self.decode_base64_to_image(value)}


class ToImage(InstanceFieldOperator):
    def process_instance_value(self, value: Any, instance: Dict[str, Any]):
        return {"image": value}
