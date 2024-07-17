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
        {"question": "What is the capital of Texas?", "answer": ""},
        {"question": "What is the color of the sky?", "answer": ""},
    ]
}
# Second, We define the prompt we show to the judge.
judge_correctness_template = InputOutputTemplate(
    instruction="Please act as an impartial judge and evaluate if the assistant's answer is correct."
    ' Answer "[[10]]" if the answer is accurate, and "[[0]]" if the answer is wrong. '
    'Please use the exact format of the verdict as "[[rate]]". '
    "You can explain your answer after the verdict"
    ".\n\n",
    input_format="[Question]\n{question}\n\n" "[Assistant's Answer]\n{answer}\n",
    output_format="[[{rating}]]",
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
llm_judge_metric = LLMAsJudge(
    inference_model=inference_model,
    template=judge_correctness_template,
    task="rating.single_turn",
    main_score=f"llm_judge_{model_name.split('/')[1].replace('-', '_')}_{platform}",
    strip_system_prompt_and_format_from_inputs=False,
)
# we wrapped all ingredients in a task card.
card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=Task(
        input_fields={"question": "str"},
        reference_fields={"answer": "str"},
        prediction_type="str",
        metrics=[llm_judge_metric],
    ),
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                instruction="Answer the following question.",
                input_format="{question}",
                output_format="{answer}",
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
