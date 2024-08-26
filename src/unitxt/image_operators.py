import re
from typing import Any, Dict

from .dict_utils import dict_get
from .operators import InstanceFieldOperator


def extract_images(text, instance):
    regex = r'<img\s+src=["\'](.*?)["\']'
    image_sources = re.findall(regex, text)
    images = []
    for image_source in image_sources:
        image = dict_get(instance, image_source)
        images.append(image)
    return images


class ImageToText(InstanceFieldOperator):
    def process_instance_value(self, value: Any, instance: Dict[str, Any]):
        if "media" not in instance:
            instance["media"] = {}
        if "images" not in instance["media"]:
            instance["media"]["images"] = []
        idx = len(instance["media"]["images"])
        instance["media"]["images"].append(value)
        return f'<img src="media/images/{idx}">'
