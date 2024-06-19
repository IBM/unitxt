from unitxt import get_logger
from unitxt.api import evaluate
from unitxt.blocks import TaskCard
from unitxt.collections_operators import Wrap
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.loaders import LoadFromDictionary
from unitxt.standard import StandardRecipe

logger = get_logger()

# Set up question answer pairs in a dictionary
data = {
    "test": [
        {"question": "What is the capital of Texas?", "answer": "Austin"},
        {"question": "What is the color of the sky?", "answer": "Blue"},
    ]
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

# Verbalize the dataset using the catalog template which adds an instructio "Answer the question.",
# and "Question:"/"Answer:" prefixes.
#
# "Answer the question.
#  Question:
#  What is the color of the sky?
#  Answer:
# "
dataset = StandardRecipe(card=card, template="templates.qa.open.title")().to_dataset()
test_dataset = dataset["test"]


# Infer using flan t5 base using HF API
model_name = "google/flan-t5-base"
inference_model = HFPipelineBasedInferenceEngine(
    model_name=model_name, max_new_tokens=32
)
# change to this to infer with IbmGenAI APIs:
#
# gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=32)
# inference_model = IbmGenAiInferenceEngine(model_name=model_name, parameters=gen_params)
#
# or to this to infer using OpenAI APIs:
#
# gen_params = OpenAiInferenceEngineParams(max_new_tokens=32)
# inference_model = OpenAiInferenceEngine(model_name=model_name, parameters=gen_params)
#
predictions = inference_model.infer(test_dataset)
dataset_with_scores = evaluate(predictions=predictions, data=test_dataset)

# Print results
for sample, prediction in zip(dataset_with_scores, predictions):
    logger.info("*" * 80)
    logger.info(f"Model input:\n{sample['source']}")
    logger.info(f"Model prediction (as returned by the model):\n{prediction}")
    logger.info(f"Model prediction (after post processing):\n{sample['prediction']}")
    logger.info(f"References:\n{sample['references']}")
    score_name = sample["score"]["instance"]["score_name"]
    score = sample["score"]["instance"]["score"]
    logger.info(f"Sample score ({score_name}) : {score}")
global_score = dataset_with_scores[0]["score"]["global"]["score"]
logger.info("*" * 80)
logger.info(f"Aggregated score ({score_name}) : {global_score}")
