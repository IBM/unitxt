from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate, TemplatesDict
from unitxt.text_utils import print_dict

logger = get_logger()

# First, we define the examples data.
data = {
    "test": [
        {
            "query": "What is the capital of Texas?",
            "document": "The capital of Texas is Austin.",
            "reference_answer": "Austin",
        },
        {
            "query": "What is the color of the sky?",
            "document": "The sky is generally black during the night.",
            "reference_answer": "Black",
        },
    ]
}
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

platform = "hf"
model_name = "meta-llama/Llama-3.2-1B"

# Infer using Llama-3.2-1B base using HF API
engine = HFPipelineBasedInferenceEngine(
    model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
)
# Change to this to infer with external APIs:
# CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]


# Third, We define the metric as LLM as a judge, with the desired platform and model.
llm_judge_metric = LLMAsJudge(
    inference_model=engine,
    template=judge_correctness_template,
    format="formats.chat_api",
    task="rating.single_turn",
    main_score=f"llm_judge_{model_name.split('/')[1].replace('-', '_')}_{platform}",
    strip_system_prompt_and_format_from_inputs=False,
)
# we wrapped all ingredients in a task card.
card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=Task(
        input_fields={"query": str, "document": str},
        reference_fields={"reference_answer": str},
        prediction_type=str,
        metrics=[llm_judge_metric],
    ),
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                instruction="Answer the following query based on the provided document.",
                input_format="Document:\n{document}\nQuery:\n{query}",
                output_format="{reference_answer}",
                postprocessors=["processors.lower_case"],
            )
        }
    ),
)

# Convert card to a dataset
dataset = load_dataset(
    card=card,
    template_card_index="simple",
    format="formats.chat_api",
    split="test",
    max_test_instances=10,
)

# Infer using Llama-3.2-1B base using HF API
engine = HFPipelineBasedInferenceEngine(
    model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
)
predictions = engine.infer(dataset)

# Evaluate the predictions using the defined metric.
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
