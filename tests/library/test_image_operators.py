import unittest
from unittest.mock import patch

from unitxt.image_operators import ToImage, extract_images
from unitxt.settings_utils import get_constants

constants = get_constants()


def create_random_jpeg_image(width, height, seed=None):
    import io

    import numpy as np
    from PIL import Image

    # Set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Create a random RGB image
    random_image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_image_array, "RGB")

    # Save the image to an in-memory bytes buffer in JPEG format
    img_byte_array = io.BytesIO()
    random_image.save(img_byte_array, format="JPEG")
    img_byte_array.seek(0)  # Rewind to the start of the stream

    # Load the JPEG image back into PIL
    return Image.open(img_byte_array)


class TestImageOperators(unittest.TestCase):
    def test_extract_images_no_images(self):
        text = "This is a text without images"
        instance = {"source": text}
        result = extract_images(instance)
        self.assertEqual(result, [])

    def test_extract_images_single_image(self):
        text = f'This is a text with <{constants.image_tag} src="image1.jpg"> image'
        instance = {"image1.jpg": "image1_data", "source": text}
        result = extract_images(instance)
        self.assertEqual(result, ["image1_data"])

    def test_extract_images_multiple_images(self):
        text = f'Text with <{constants.image_tag} src="image1.jpg"> and <{constants.image_tag} src="image2.png">'
        instance = {
            "image1.jpg": "image1_data",
            "image2.png": "image2_data",
            "source": text,
        }
        result = extract_images(instance)
        self.assertEqual(result, ["image1_data", "image2_data"])

    def test_extract_images_missing_image(self):
        text = f'Text with <{constants.image_tag} src="image1.jpg"> and <{constants.image_tag} src="missing.png">'
        instance = {"image1.jpg": "image1_data", "source": text}
        with self.assertRaises(ValueError):
            extract_images(instance)

    @patch("unitxt.dict_utils.dict_get")
    def test_extract_images_dict_get_raises_value_error(self, mock_dict_get):
        text = f'Text with <{constants.image_tag} src="missing.png">'
        instance = {"source": text}
        mock_dict_get.side_effect = ValueError("Key not found")
        with self.assertRaises(ValueError):
            extract_images(instance)


class TestImageToText(unittest.TestCase):
    def setUp(self):
        self.operator = ToImage(field="dummy")

    def test_process_instance_value_new_media(self):
        instance = {}
        value = create_random_jpeg_image(10, 10, 1)
        result = self.operator.process_instance_value(value, instance)
        self.assertEqual(result, {"image": value, "format": "JPEG"})

    # def test_process_instance_value_existing_media(self):
    #     instance = {"media": {"images": ["existing_image"]}}
    #     value = "new_image_data"
    #     result = self.operator.process_instance_value(value, instance)
    #     self.assertEqual(result, '<img src="media/images/1">')
    #     self.assertEqual(
    #         instance, {"media": {"images": ["existing_image", "new_image_data"]}}
    #     )

    # def test_process_instance_value_multiple_calls(self):
    #     instance = {}
    #     values = ["image1", "image2", "image3"]
    #     results = []
    #     for value in values:
    #         results.append(self.operator.process_instance_value(value, instance))
    #     self.assertEqual(
    #         results,
    #         [
    #             '<img src="media/images/0">',
    #             '<img src="media/images/1">',
    #             '<img src="media/images/2">',
    #         ],
    #     )
    #     self.assertEqual(
    #         instance, {"media": {"images": ["image1", "image2", "image3"]}}
    #     )
