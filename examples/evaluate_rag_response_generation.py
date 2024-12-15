from unitxt.api import evaluate, load_dataset
from unitxt.blocks import (
    TaskCard,
)
from unitxt.collections_operators import Wrap
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.loaders import LoadFromDictionary
from unitxt.operators import Rename, Set
from unitxt.templates import MultiReferenceTemplate, TemplatesDict
from unitxt.text_utils import print_dict

# Assume the RAG data is proved in this format
data = {
    "test": [
        {
            "query": "What city is the largest in Texas?",
            "extracted_chunks": "Austin is the capital of Texas.\nHouston is the the largest city in Texas but not the capital of it. ",
            "expected_answer": "Houston",
        },
        {
            "query": "What city is the capital of Texas?",
            "extracted_chunks": "Houston is the the largest city in Texas but not the capital of it. ",
            "expected_answer": "Austin",
        },
    ]
}


card = TaskCard(
    # Assumes this csv, contains 3 fields
    # question (string), extracted_chunks (string), expected_answer (string)
    loader=LoadFromDictionary(data=data),
    # Map these fields to the fields of the task.rag.response_generation task.
    # See https://www.unitxt.ai/en/latest/catalog/catalog.tasks.rag.response_generation.html
    preprocess_steps=[
        Rename(field_to_field={"query": "question"}),
        Wrap(field="extracted_chunks", inside="list", to_field="contexts"),
        Wrap(field="expected_answer", inside="list", to_field="reference_answers"),
        Set(
            fields={
                "contexts_ids": [],
            }
        ),
    ],
    # Specify the task and the desired metrics (note that these are part of the default
    # metrics for the task, so the metrics selection can be omitted).
    task="tasks.rag.response_generation",
    # Specify a default template
    templates=TemplatesDict(
        {
            "simple": MultiReferenceTemplate(
                instruction="Answer the question based on the information provided in the document given below.\n\n",
                input_format="Document: {contexts}\nQuestion: {question}",
                references_field="reference_answers",
            ),
        }
    ),
)

# Verbalize the dataset using the template
dataset = load_dataset(
    card=card,
    template_card_index="simple",
    format="formats.chat_api",
    split="test",
    max_test_instances=10,
)


# Infer using Llama-3.2-1B base using HF API
engine = HFPipelineBasedInferenceEngine(
    model_name="meta-llama/Llama-3.2-1B", max_new_tokens=32
)
# Change to this to infer with external APIs:
# CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]

predictions = engine.infer(dataset)
evaluated_dataset = evaluate(predictions=predictions, data=dataset)

# Print results
for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ],
    )
