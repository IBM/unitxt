import json

from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.templates import SpanLabelingTemplate

logger = get_logger()
classes = ["Person", "Location", "Organization"]


test_set = [
    {
        "text": "John lives in Texas.",
        "classes": classes,
        "spans_starts": [0, 14],
        "spans_ends": [5, 19],
        "labels": ["Person", "Location"],
    },
    {
        "text": "Phil works at Apple and eats an apple.",
        "classes": classes,
        "spans_starts": [0, 14],
        "spans_ends": [5, 19],
        "labels": ["Person", "Organization"],
    },
]


template = SpanLabelingTemplate(
    instruction="""From the following text, extract the  entities of one of the following entity types: {classes}.
Return the output in this exact format:
The output should be a comma separated list of pairs of entity and corresponding entity_type.
Use a colon to separate between the entity and entity_type. """,
    input_format="{text_type}:\n{text}",
    postprocessors=["processors.to_span_label_pairs"],
)

dataset = create_dataset(
    task="tasks.span_labeling.extraction",
    test_set=test_set,
    template=template,
    split="test",
    format="formats.chat_api",
)

# Infer using Llama-3.2-1B base using HF API
# model = HFPipelineBasedInferenceEngine(
#   model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
# )
# Change to this to infer with external APIs:

model = CrossProviderInferenceEngine(model="llama-3-8b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]


predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Example prompt:")

print(json.dumps(results.instance_scores[0]["source"], indent=4))

print("Instance Results:")
print(
    results.instance_scores.to_df(
        columns=[
            "text",
            "prediction",
            "processed_prediction",
            "processed_references",
            "score",
            "score_name",
        ]
    ).to_markdown()
)
