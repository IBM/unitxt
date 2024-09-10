import unittest
from unittest.mock import patch

from unitxt.image_operators import NormalizeImage, extract_images


class TestImageOperators(unittest.TestCase):
    def test_extract_images_no_images(self):
        text = "This is a text without images"
        instance = {}
        result = extract_images(text, instance)
        self.assertEqual(result, [])

    def test_extract_images_single_image(self):
        text = 'This is a text with <img src="image1.jpg"> image'
        instance = {"image1.jpg": "image1_data"}
        result = extract_images(text, instance)
        self.assertEqual(result, ["image1_data"])

    def test_extract_images_multiple_images(self):
        text = 'Text with <img src="image1.jpg"> and <img src="image2.png">'
        instance = {"image1.jpg": "image1_data", "image2.png": "image2_data"}
        result = extract_images(text, instance)
        self.assertEqual(result, ["image1_data", "image2_data"])

    def test_extract_images_missing_image(self):
        text = 'Text with <img src="image1.jpg"> and <img src="missing.png">'
        instance = {"image1.jpg": "image1_data"}
        with self.assertRaises(ValueError):
            extract_images(text, instance)

    @patch("unitxt.dict_utils.dict_get")
    def test_extract_images_dict_get_raises_value_error(self, mock_dict_get):
        text = 'Text with <img src="missing.png">'
        instance = {}
        mock_dict_get.side_effect = ValueError("Key not found")
        with self.assertRaises(ValueError):
            extract_images(text, instance)


class TestImageToText(unittest.TestCase):
    def setUp(self):
        self.operator = NormalizeImage(field="dummy")

    def test_process_instance_value_new_media(self):
        instance = {}
        value = "image_data"
        result = self.operator.process_instance_value(value, instance)
        self.assertEqual(result, '<img src="media/images/0">')
        self.assertEqual(instance, {"media": {"images": ["image_data"]}})

    def test_process_instance_value_existing_media(self):
        instance = {"media": {"images": ["existing_image"]}}
        value = "new_image_data"
        result = self.operator.process_instance_value(value, instance)
        self.assertEqual(result, '<img src="media/images/1">')
        self.assertEqual(
            instance, {"media": {"images": ["existing_image", "new_image_data"]}}
        )

    def test_process_instance_value_multiple_calls(self):
        instance = {}
        values = ["image1", "image2", "image3"]
        results = []
        for value in values:
            results.append(self.operator.process_instance_value(value, instance))
        self.assertEqual(
            results,
            [
                '<img src="media/images/0">',
                '<img src="media/images/1">',
                '<img src="media/images/2">',
            ],
        )
        self.assertEqual(
            instance, {"media": {"images": ["image1", "image2", "image3"]}}
        )


if __name__ == "__main__":
    unittest.main()
