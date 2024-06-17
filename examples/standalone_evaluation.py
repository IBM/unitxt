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

data = {
    "test": [
        {"question": "What is the capital of Texas?", "answer": "Austin"},
        {"question": "What is the color of the sky?", "answer": "Blue"},
    ]
}

card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=Task(
        inputs={"question": "str"},
        outputs={"answer": "str"},
        prediction_type="str",
        metrics=["metrics.accuracy"],
    ),
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

dataset = StandardRecipe(card=card, template_card_index="simple")().to_dataset()
test_dataset = dataset["test"]


model_name = "google/flan-t5-base"
inference_model = HFPipelineBasedInferenceEngine(
    model_name=model_name, max_new_tokens=32
)

# gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=32)
# inference_model = IbmGenAiInferenceEngine(model_name=model_name, parameters=gen_params)
predictions = inference_model.infer(test_dataset)
dataset_with_scores = evaluate(predictions=predictions, data=test_dataset)

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
