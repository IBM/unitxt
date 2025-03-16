import json

from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

logger = get_logger()
keys = ["Worker", "LivesIn", "WorksAt"]


def text_to_image(text: str):
    """Return a image with the input text render in it."""
    from PIL import Image, ImageDraw, ImageFont

    bg_color = (255, 255, 255)
    text_color = (0, 0, 0)
    font_size = 10
    font = ImageFont.load_default(size=font_size)

    img = Image.new("RGB", (1, 1), bg_color)

    # Get dimensions of the text
    # text_width, text_height = font.getsize_multiline(value)

    # Create a new image with appropriate size
    img = Image.new("RGB", (1000, 1000), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw the text on the image
    draw.multiline_text((0, 0), text, fill=text_color, font=font)
    return {"image": img, "format": "png"}


test_set = [
    {
        "input": text_to_image("John lives in New York."),
        "keys": keys,
        "key_value_pairs_answer": {"Worker": "John", "LivesIn": "New York"},
    },
    {
        "input": text_to_image("Phil works at Apple and eats an apple."),
        "keys": keys,
        "key_value_pairs_answer": {"Worker": "Phil", "WorksAt": "Apple"},
    },
]


dataset = create_dataset(
    task="tasks.key_value_extraction",
    template="templates.key_value_extraction.extract_in_json_format",
    test_set=test_set,
    split="test",
    format="formats.chat_api",
    metrics=["metrics.key_value_extraction.accuracy","metrics.key_value_extraction.token_overlap"]
)

model = CrossProviderInferenceEngine(
    model="llama-3-2-90b-vision-instruct", provider="watsonx"
)

predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Example prompt:")

print(json.dumps(results.instance_scores[0]["source"], indent=4))

print("Instance Results:")
print(results.instance_scores)

print("Global Results:")
print(results.global_scores.summary)
