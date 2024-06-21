from unitxt import get_logger
from unitxt.api import evaluate
from unitxt.blocks import Task, TaskCard
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.loaders import LoadFromDictionary
from unitxt.standard import StandardRecipe
from unitxt.templates import InputOutputTemplate, TemplatesDict

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
    # Define the QA task input and output and metrics.
    task=Task(
        inputs={"question": "str"},
        outputs={"answer": "str"},
        prediction_type="str",
        metrics=["metrics.accuracy"],
    ),
    # Create a simple template that formats the input.
    # Add lowercase normalization as a post processor.
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                instruction="Answer the following question.",
                input_format="{question}",
                output_format="{answer}",
                postprocessors=["processors.lower_case"],
            )
        }
    ),
)

# Verbalize the dataset using the template
dataset = StandardRecipe(card=card, template_card_index="simple")().to_dataset()
test_dataset = dataset["test"]


# Infere using flan t5 base using HF API
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
# gen_params = IOpenAiInferenceEngineParams(max_new_tokens=32)
# inference_model = OpenAiInferenceEngine(model_name=model_name, parameters=gen_params)
#
# Note that to run with OpenAI APIs you need to change the loader specification, to
# define that your data can be sent to a public API:
#
# loader=LoadFromDictionary(data=data,data_classification_policy=["public"]),

predictions = inference_model.infer(test_dataset)
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

# Print results
for instance in evaluated_dataset:
    logger.info("*" * 80)
    logger.info(f"Model input:\n{instance['source']}")
    logger.info(
        f"Model prediction (as returned by the model):\n{instance['prediction']}"
    )
    logger.info(
        f"Model prediction (after post processing):\n{instance['processed_prediction']}"
    )
    logger.info(f"References:\n{instance['references']}")
    logger.info(
        f"References (after post processing):\n{instance['processed_references']}"
    )
    score_name = instance["score"]["instance"]["score_name"]
    score = instance["score"]["instance"]["score"]
    logger.info(f"Sample score ({score_name}) : {score}")
global_score = evaluated_dataset[0]["score"]["global"]["score"]
logger.info("*" * 80)
logger.info(f"Aggregated score ({score_name}) : {global_score}")
