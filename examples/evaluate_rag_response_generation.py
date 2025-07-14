from unitxt.api import evaluate, load_dataset
from unitxt.blocks import (
    TaskCard,
)
from unitxt.collections_operators import Wrap
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.loaders import LoadFromDictionary
from unitxt.operators import Rename, Set
from unitxt.templates import MultiReferenceTemplate, TemplatesDict

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

# select recommended metrics according to your available resources.
metrics = [
    "metrics.rag.response_generation.recommended.cpu_only.all",
    # "metrics.rag.response_generation.recommended.small_llm.all",
    # "metrics.rag.response_generation.recommended.llmaj_watsonx.all",
    # "metrics.rag.response_generation.recommended.llmaj_rits.all"
    # "metrics.rag.response_generation.recommended.llmaj_azure.all"
]

# Verbalize the dataset using the template
dataset = load_dataset(
    card=card,
    template_card_index="simple",
    format="formats.chat_api",
    split="test",
    max_test_instances=10,
    metrics=metrics,
)


model = CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "hf-local"]

predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Instance Results:")
print(results.instance_scores.summary)
