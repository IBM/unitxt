import ast
import json
from typing import Any, Dict, List, Tuple

from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.blocks import Task
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.metrics import CustomF1
from unitxt.operators import FieldOperator
from unitxt.processors import PostProcess
from unitxt.templates import InputOutputTemplate

logger = get_logger()
keys = ["Worker", "LivesIn", "WorksAt"]


def text_to_image(value: str):
    from PIL import Image, ImageDraw, ImageFont

    bg_color = (255, 255, 255)
    text_color = (0, 0, 0)
    font_size = 10
    font = ImageFont.truetype("Helvetica", font_size)

    img = Image.new("RGB", (1, 1), bg_color)

    # Get dimensions of the text
    # text_width, text_height = font.getsize_multiline(value)

    # Create a new image with appropriate size
    img = Image.new("RGB", (1000, 1000), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw the text on the image
    draw.multiline_text((0, 0), value, fill=text_color, font=font)
    return {"image": img, "format": "png"}


test_set = [
    {
        "input": text_to_image("John lives in Texas."),
        "keys": keys,
        "key_value_pairs_answer": {"Worker": "John", "LivesIn": "Texas"},
    },
    {
        "input": text_to_image("Phil works at Apple and eats an apple."),
        "keys": keys,
        "key_value_pairs_answer": {"Worker": "Phil", "WorksAt": "Apple"},
    },
]


class JsonStrToListOfKeyValuePairs(FieldOperator):
    def process_value(self, text: str) -> List[Tuple[str, str]]:
        text = text.replace("null", "None")

        try:
            dict_value = ast.literal_eval(text)  # json.loads(text)
        except:
            print("Unable to load dict value from:", text)
            dict_value = {}
        return [(key, value) for key, value in dict_value.items() if value is not None]


template = InputOutputTemplate(
    instruction="Extract the key value pairs from the input. Return a valid json object with the following keys: {keys}. Return only the json representation, no additional text or explanations.",
    input_format="{input}",
    output_format="{key_value_pairs_answer}",
    postprocessors=[PostProcess(JsonStrToListOfKeyValuePairs())],
)


class KeyValueExtraction(CustomF1):
    """F1 Metrics that receives as input a list of (Key,Value) pairs."""

    prediction_type = List[Tuple[str, str]]

    def get_element_group(self, element, additional_input):
        return element[0]

    def get_element_representation(self, element, additional_input):
        return str(element)


task = Task(
    __description__="This is a key value extraction task, where a specific list of possible 'keys' need to be extracted from the input.  The ground truth is provided key-value pairs in the form of the dictionary.  The results are evaluating using F1 score metric, that expects the predictions to be converted into a list of (key,value) pairs. ",
    input_fields={"input": Any, "keys": List[str]},
    reference_fields={"key_value_pairs_answer": Dict[str, str]},
    prediction_type=List[Tuple[str, str]],
    metrics=[KeyValueExtraction()],
)

dataset = create_dataset(
    task=task,
    template=template,
    test_set=test_set,
    split="test",
    format="formats.chat_api",
)

# Infer using Llama-3.2-1B base using HF API
# model = HFPipelineBasedInferenceEngine(
#   model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
# )
# Change to this to infer with external APIs:

model = CrossProviderInferenceEngine(
    model="llama-3-2-11b-vision-instruct", provider="watsonx"
)
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]


predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Example prompt:")

print(json.dumps(results.instance_scores[0]["source"], indent=4))

print("Instance Results:")
print(results.instance_scores)
