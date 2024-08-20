from unitxt import get_logger, get_settings, load_dataset
from unitxt.api import evaluate
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.text_utils import print_dict

logger = get_logger()
settings = get_settings()
settings.allow_unverified_code = True

# Use the HF load_dataset API, to load the squad QA dataset using the standard template in the catalog.
# We set loader_limit to 20 to reduce download time.
dataset = load_dataset(
    card="cards.squad",
    template="templates.qa.with_context.simple",
    metrics=[
        "metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn"
    ],
    loader_limit=20,
)
test_dataset = dataset["test"]

# Infer a model to get predictions.
model_name = "google/flan-t5-base"
inference_model = HFPipelineBasedInferenceEngine(
    model_name=model_name, max_new_tokens=32
)
predictions = inference_model.infer(test_dataset)

# Evaluate the predictions using the defined metric.
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
