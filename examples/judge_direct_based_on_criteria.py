from unitxt.api import evaluate, load_dataset
from unitxt.blocks import (
    TaskCard,
)
from unitxt.collections_operators import Wrap
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.llm_as_judge import LLMJudgeDirect
from unitxt.llm_as_judge_constants import (
    CriteriaWithOptions,
)
from unitxt.loaders import LoadFromDictionary
from unitxt.operators import Rename, Set
from unitxt.templates import MultiReferenceTemplate, TemplatesDict

# First, describe a judgement criteria
description = """
Evaluate the response based on two criteria: adherence to instructions and compliance with the specified format.
Adherence to Instructions: Does the response fully follow the provided instruction?
Compliance with Format: Does the response align with the requested structure, style, or format (e.g., bullet points, headings, specific phrasing)?
"""

criteria = CriteriaWithOptions.from_obj(
    {
        "name": "test for adherence and compliance",
        "description": description,
        "options": [
            {"name": "Excellent", "description": ""},
            {"name": "Good", "description": ""},
            {"name": "mediocre", "description": ""},
            {"name": "Bad", "description": ""},
            {"name": "Very Bad", "description": ""},
        ],
        "option_map": {
            "Very Good": 1.0,
            "Good": 0.75,
            "mediocre": 0.5,
            "Bad": 0.25,
            "Very Bad": 0,
        },
    }
)

# now = define the judge metric using the criteria
metric = LLMJudgeDirect(
    inference_engine=CrossProviderInferenceEngine(  # or your favorite inference model
        model="llama-3-1-70b-instruct", max_tokens=1024
    ),
    criteria=criteria,
    # the fields from the generation task to be presented to the judge. Those fields must be present
    # in the generation task so they can be embedded here
    context_fields=["question", "reference_answers", "metadata/template/instruction"],
    criteria_field="criteria",
    generate_summaries=False,
    check_positional_bias=False,
)

# now we can use this metric, the same way as any other metric
# (as long as the criteria fields are present at the generation task)


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


dataset = load_dataset(
    card=card,
    template_card_index="simple",
    format="formats.chat_api",
    split="test",
    max_test_instances=10,
    metrics=[metric],
)

predictions = ["Austin" "Austin"]

results = evaluate(predictions=predictions, data=dataset)

print("Global Scores:")
print(results.global_scores.summary)

print("Instance Scores:")
print(results.instance_scores.summary)
