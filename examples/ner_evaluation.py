import json

from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

logger = get_logger()
entity_types = ["Person", "Location", "Organization"]


test_set = [
    {
        "text": "John lives in Texas.",
        "entity_types": entity_types,
        "spans_starts": [0, 14],
        "spans_ends": [5, 19],
        "labels": ["Person", "Location"],
    },
    {
        "text": "Phil works at Apple and eats an apple.",
        "entity_types": entity_types,
        "spans_starts": [0, 14],
        "spans_ends": [5, 19],
        "labels": ["Person", "Organization"],
    },
]


dataset = create_dataset(
    task="tasks.ner.all_entity_types",
    test_set=test_set,
    split="test",
    format="formats.chat_api",
)

# Infer using SmolLM2 using HF API
# model = HFPipelineBasedInferenceEngine(
#   model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_new_tokens=32
# )
# Change to this to infer with external APIs:

model = CrossProviderInferenceEngine(model="llama-3-3-70b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]


predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Example prompt:")

print(json.dumps(results.instance_scores[0]["source"], indent=4))

print("Instance Results:")
print(
    results.instance_scores.to_markdown(
        columns=[
            "text",
            "prediction",
            "processed_prediction",
            "processed_references",
            "score",
            "score_name",
        ]
    )
)
