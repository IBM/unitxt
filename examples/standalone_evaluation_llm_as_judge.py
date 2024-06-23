from unitxt import get_logger
from unitxt.api import evaluate
from unitxt.blocks import Task, TaskCard
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.loaders import LoadFromDictionary
from unitxt.standard import StandardRecipe
from unitxt.templates import InputOutputTemplate, TemplatesDict

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
    instruction="Please act as an impartial judge and evaluate if the answer of the assistant is correct."
    "Rate the response on a scale of 1 to 10, where 1 means totally wrong, and 10 means totally correct,"
    ' by strictly following this format: "[[rating]]"'
    ".\n\n",
    input_format="[Question]\n{question}\n\n" "[Assistant's Answer]\n{answer}\n",
    output_format="[[{rating}]]",
    postprocessors=[
        r"processors.extract_mt_bench_rating_judgment",
    ],
)
# Third, We define the metric as LLM as a judge, with the desired platform and model.
llm_judge_metric = LLMAsJudge(
    inference_model=HFPipelineBasedInferenceEngine(
        model_name="google/flan-t5-large", max_new_tokens=256, use_fp16=True
    ),
    template=judge_correctness_template,
    task="rating.single_turn",
    main_score="flan-t5-large_huggingface",
    strip_system_prompt_and_format_from_inputs=False,
)
# we wrapped all ingredients in a task card.
card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=Task(
        inputs={"question": "str"},
        outputs={"answer": "str"},
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
recipe = StandardRecipe(card=card, template_card_index="simple")
stream = recipe()
dataset = stream.to_dataset()
test_dataset = dataset["test"]

# Infer a model to get predictions.
model_name = "google/flan-t5-base"
inference_model = HFPipelineBasedInferenceEngine(
    model_name=model_name, max_new_tokens=32
)
predictions = inference_model.infer(test_dataset)

# Evaluate the predictions using the defined metric.
dataset_with_scores = evaluate(predictions=predictions, data=test_dataset)

for sample, prediction in zip(dataset_with_scores, predictions):
    logger.info("*" * 80)
    logger.info(f"Model input:\n{sample['source']}")
    logger.info(f"Model prediction (as returned by the model):\n{prediction}")
    logger.info(f"Model prediction (after post processing):\n{sample['prediction']}")
    logger.info(f"References:\n{sample['references']}")
    score_name = sample["score"]["instance"]["score_name"]
    score = sample["score"]["instance"]["score"]
    logger.info(f"Sample score ({score_name}) : {score}")
global_score = dataset_with_scores[0]["score"]["global"]["score"]
logger.info("*" * 80)
logger.info(f"Aggregated score ({score_name}) : {global_score}")
