from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.blocks import Task
from unitxt.inference import (
    CrossProviderInferenceEngine,
    HFPipelineBasedInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.templates import InputOutputTemplate

logger = get_logger()

# First, we define the examples data we want to evaluate using LLM as judge.
data = [
    {
        "query": "What is the capital of Texas?",
        "document": "The capital of Texas is Austin.",
        "reference_answer": "Austin",
    },
    {
        "query": "What is the color of the sky right now?",
        "document": "The sky is generally black during the night.",
        "reference_answer": "Black",
    },
]

# Second, We define the prompt we show to the judge.
#
# Note that "question" is the full input provided to the original model, and "answer" is the original model
# output.  For example , this is sample input provided to the LLM as judge model.
#
# Please act as an impartial judge and evaluate if the assistant's answer is correct. Answer "[[10]]" if the answer is accurate, and "[[0]]" if the answer is wrong. Please use the exact format of the verdict as "[[rate]]".
# You can explain your answer after the verdict.
# [User's input]
# Answer the following query based on the provided document.
# Document:
# The sky is generally black during the night.
# Query:
# What is the color of the sky?
#
# [Assistant's Answer]
# black

judge_correctness_template = InputOutputTemplate(
    instruction="Please act as an impartial judge and evaluate if the assistant's answer is correct."
    ' Answer "[[10]]" if the answer is accurate, and "[[0]]" if the answer is wrong. '
    'Please use the exact format of the verdict as "[[rate]]". '
    "You can explain your answer after the verdict"
    ".\n\n",
    input_format="[User's input]\n{question}\n" "[Assistant's Answer]\n{answer}\n",
    output_format="[[{rating}]]",
    postprocessors=[
        r"processors.extract_mt_bench_rating_judgment",
    ],
)

# Third, We define the metric as LLM as a judge, with the desired platform and model.
llm_judge_metric = LLMAsJudge(
    inference_model=CrossProviderInferenceEngine(
        model="llama-3-8b-instruct", max_tokens=1024, provider="watsonx"
    ),
    template=judge_correctness_template,
    format="formats.chat_api",
    task="rating.single_turn",
    main_score="llm_judge_score",
    strip_system_prompt_and_format_from_inputs=False,
)

task = Task(
    input_fields={"query": str, "document": str},
    reference_fields={"reference_answer": str},
    prediction_type=str,
    metrics=[llm_judge_metric],
)

template = InputOutputTemplate(
    instruction="Answer the following query based on the provided document.",
    input_format="Document:\n{document}\nQuery:\n{query}",
    output_format="{reference_answer}",
    postprocessors=["processors.lower_case"],
)

dataset = create_dataset(
    test_set=data,
    task=task,
    template=template,
    format="formats.chat_api",
    split="test",
    max_test_instances=10,
)

# Infer using Llama-3.2-1B base using HF API
model = HFPipelineBasedInferenceEngine(
    model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
)
predictions = model(dataset)

# Evaluate the predictions using the defined metric.
results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Instance Results:")
print(results.instance_scores.summary)
