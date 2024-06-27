from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import TaskCard
from unitxt.collections_operators import Wrap
from unitxt.inference import IbmGenAiInferenceEngine, IbmGenAiInferenceEngineParams
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate
from unitxt.text_utils import print_dict

logger = get_logger()

# Set up question answer pairs in a dictionary
data = {
    "train": [
        {"question": "How many days in a week", "answer": "7"},
        {"question": "Where is Spain?", "answer": "Europe"},
        {"question": "When was IBM founded?", "answer": "1911"},
        {"question": "Can pigs fly?", "answer": "No"},
    ],
    "test": [
        {"question": "What is the capital of Texas?", "answer": "Austin"},
        {"question": "What is the color of the sky?", "answer": "Blue"},
    ],
}

card = TaskCard(
    # Load the data from the dictionary.  Data can be  also loaded from HF, CSV files, COS and other sources
    # with different loaders.
    loader=LoadFromDictionary(data=data),
    # Use the standard open qa QA task input and output and metrics.
    # It has "question" input field and "answers" output field.
    # The default evaluation metric used is rouge.
    task="tasks.qa.open",
    # Because the standand QA tasks supports multiple references in the "answers" field,
    # we wrap the raw dataset's "answer" field in a list and store in a the "answers" field.
    preprocess_steps=[Wrap(field="answer", inside="list", to_field="answers")],
)

template = InputOutputTemplate(
    instruction="Answer the following questions in one word.",
    input_format="{question}",
    output_format="{answers}",
    postprocessors=["processors.lower_case"],
)

dataset = load_dataset(
    card=card,
    template=template,
    format="formats.llama3_chat",
    num_demos=2,
    demos_pool_size=3,
)
test_dataset = dataset["test"]

model_name = "meta-llama/llama-3-70b-instruct"
gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=32)
inference_model = IbmGenAiInferenceEngine(model_name=model_name, parameters=gen_params)

predictions = inference_model.infer(test_dataset)
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

print_dict(
    evaluated_dataset[0],
    keys_to_print=[
        "source",
        "prediction",
        "processed_prediction",
        "references",
        "score",
    ],
)
