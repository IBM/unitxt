from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.eval_assist_constants import (
    CriteriaWithOptions,
    EvaluatorNameEnum,
)
from unitxt.eval_assist_llm_as_judge_direct import EvalAssistLLMAsJudgeDirect
from unitxt.inference import LiteLLMInferenceEngine
from unitxt.text_utils import print_dict

logger = get_logger()

criteria = CriteriaWithOptions.from_jsons(s="""
{
    "name": "Temperature in Fahrenheit and Celsius",
    "description": "In the response, if there is a numerical temperature present, is it denominated in both Fahrenheit and Celsius?",
    "options": [
        {
            "name": "Yes",
            "description": "The temperature reading is provided in both Fahrenheit and Celsius."
        },
        {
            "name": "No",
            "description": "The temperature reading is provided either in Fahrenheit or Celsius, but not both."
        },
        {
            "name": "Pass",
            "description": "There is no numerical temperature reading in the response."
        }
    ],
    "option_map": {"Yes": 1.0, "No": 0.5, "Pass": 0.0}
}
""")


data = [
    {"question": "How is the weather?"},
    {"question": "How is the weather?"},
    {"question": "How is the weather?"},
]

metric = EvalAssistLLMAsJudgeDirect(
    inference_engine=LiteLLMInferenceEngine(
        model="watsonx/meta-llama/llama-3-1-70b-instruct", max_tokens=1024
    ),
    evaluator_name=EvaluatorNameEnum.LLAMA3_1_70B.name,
    criteria=criteria,
    context_fields=["question"],
    criteria_field="criteria"
)

test_dataset = create_dataset(
    task="tasks.qa.open", test_set=data, metrics=[metric], split="test"
)

predictions = [
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
]

evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "source",
            "prediction",
            # "processed_prediction",
            # "references",
            "score",
        ],
    )
