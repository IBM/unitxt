import json

from unitxt import get_logger, get_settings, load_dataset
from unitxt.api import evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.templates import NullTemplate

logger = get_logger()
settings = get_settings()

test_instances = 10

# Use the HF load_dataset API, to load the squad QA dataset using the standard template in the catalog.
# We set loader_limit to 20 to reduce download time.

dataset = load_dataset(
    card="cards.squad",
    loader_limit=test_instances,
    max_test_instances=test_instances,
    split="test",
)

# Infer a model to get predictions.
inference_model_1 = CrossProviderInferenceEngine(
    model="llama-3-2-1b-instruct", provider="watsonx"
)

inference_model_2 = CrossProviderInferenceEngine(
    model="llama-3-8b-instruct", provider="watsonx"
)

inference_model_3 = CrossProviderInferenceEngine(
    model="llama-3-70b-instruct", provider="watsonx"
)

"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""
predictions_1 = inference_model_1.infer(dataset)
predictions_2 = inference_model_2.infer(dataset)
predictions_3 = inference_model_3.infer(dataset)

gold_answers = [d[0] for d in dataset["references"]]

# Evaluate the predictions using the defined metric.
predictions = [
    list(t)
    for t in list(zip(gold_answers, predictions_1, predictions_2, predictions_3))
]

print(json.dumps(predictions, indent=4))

criterias = ["factually_consistent"]
metrics = [
    "metrics.llm_as_judge.pairwise.rits.llama3_1_405b"
    f"[criteria=metrics.llm_as_judge.pairwise.criterias.{criteria},"
    "context_fields=[context,question]]"
    for criteria in criterias
]
dataset = load_dataset(
    card="cards.squad",
    loader_limit=test_instances,
    max_test_instances=test_instances,
    metrics=metrics,
    template=NullTemplate(),
    split="test",
)

evaluated_predictions = evaluate(predictions=predictions, data=dataset)

prediction_scores_by_system = {
    f"system_{system}": {
        "per_instance_winrate": [
            instance["score"]["instance"][f"{system}_winrate"]
            for instance in evaluated_predictions
        ],
        "mean_winrate": evaluated_predictions[0]["score"]["global"][
            f"{system}_winrate"
        ],
    }
    for system in range(1, len(predictions[0]) + 1)
}
print(json.dumps(prediction_scores_by_system, indent=4))