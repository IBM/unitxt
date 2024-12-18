from typing import List
from unitxt import get_logger
from unitxt.api import create_dataset, evaluate, load_dataset
from unitxt.card import TaskCard, Task
from unitxt.eval_assist_constants import (
    Criteria,
    EvaluatorNameEnum,
)
from unitxt.eval_assist_llm_as_judge_pairwise import EvalAssistLLMAsJudgePairwise
from unitxt.inference import LiteLLMInferenceEngine
from unitxt.loaders import LoadFromDictionary
from unitxt.text_utils import print_dict

logger = get_logger()

criteria = Criteria.from_jsons(s="""
{
    "name": "Temperature in Fahrenheit and Celsius",
    "description": "The temperature is described in both Fahrenheit and Celsius."
}
""")

print(criteria)

data = {
    "test": [
        {
            "question": "How is the weather?",
        },
        {
            "question": "Tell me a joke about cats",
        },
    ]
}

metric = EvalAssistLLMAsJudgePairwise(
    inference_engine=LiteLLMInferenceEngine(
        model="watsonx/meta-llama/llama-3-1-70b-instruct", max_tokens=1024
    ),
    evaluator_name=EvaluatorNameEnum.LLAMA3_1_70B.name,
    criteria=criteria,
    context_fields=["question"],
)


card = TaskCard(
    loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
    task=Task(
        input_fields={"question": str},
        reference_fields={},
        prediction_type=List[str],
        metrics=[metric],
    ),
)

predictions = [
    [
        """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
        """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
        """On most days, the weather is warm and humid. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    ],
    [
        """Why did the cat cross the road? To cat to the other side.""",
        """Why did the cat sit on the computer? Because it wanted to keep an eye on the mouse!""",
    ],
]

test_dataset = load_dataset(card=card, template="templates.empty")["test"]

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
