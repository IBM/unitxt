from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.llm_as_judge import ScoreLLMAsJudge
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
    input_format="[User's input]\n{model_input}\n"
    "[Assistant's Answer]\n{model_output}\n",
    output_format="[[{score}]]",
    postprocessors=[
        r"processors.extract_mt_bench_rating_judgment",
    ],
)

platform = "hf"
model_name = "google/flan-t5-large"
inference_model = HFPipelineBasedInferenceEngine(
    model_name=model_name, max_new_tokens=256, use_fp16=True
)
# change to this to infer with IbmGenAI APIs:
#
# platform = 'ibm_gen_ai'
# model_name = 'meta-llama/llama-3-70b-instruct'
# inference_model = IbmGenAiInferenceEngine(model_name="meta-llama/llama-3-70b-instruct", max_new_tokens=32)


# Third, We define the metric as LLM as a judge, with the desired platform and model.
llm_judge_metric = ScoreLLMAsJudge(
    inference_model=inference_model,
    template=judge_correctness_template,
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
dataset = load_dataset(card=card, template_card_index="simple")
test_dataset = dataset["test"]

# Infer a model to get predictions.
model_name = "google/flan-t5-base"
inference_model = HFPipelineBasedInferenceEngine(
    model_name=model_name, max_new_tokens=32
)
predictions = inference_model.infer(test_dataset)

# Evaluate the predictions using the defined metric.
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

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
