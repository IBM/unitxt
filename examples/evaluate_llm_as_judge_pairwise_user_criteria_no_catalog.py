from typing import Any, List

from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.card import Task, TaskCard
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.llm_as_judge import CreateCriteriaFromDict, LLMJudgePairwise
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import NullTemplate

logger = get_logger()

temperature_criteria_json = {
    "name": "Temperature in Fahrenheit and Celsius",
    "description": "The temperature is described in both Fahrenheit and Celsius.",
}


funny_criteria_json = {"name": "Funny joke", "description": "Is the response funny?"}

data = {
    "test": [
        {"question": "How is the weather?", "judgement": temperature_criteria_json},
        {"question": "Tell me a joke about cats", "judgement": funny_criteria_json},
    ]
}

metric = LLMJudgePairwise(
    inference_engine=CrossProviderInferenceEngine(
        model="llama-3-1-70b-instruct", max_tokens=1024
    ),
    context_fields=["question"],
    criteria_field="criteria",
)


card = TaskCard(
    loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
    preprocess_steps=[
        CreateCriteriaFromDict(field="judgement", to_field="criteria"),
    ],
    task=Task(
        input_fields={"question": str},
        reference_fields={"criteria": Any},
        prediction_type=List[str],
        metrics=[metric],
        default_template=NullTemplate(),
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
        """What is red, yellow and green? A traffic light.""",
    ],
]

dataset = load_dataset(card=card, split="test")

results = evaluate(predictions=predictions, data=dataset)

print("Global Scores:")
print(results.global_scores.summary)

print("Instance Scores:")
print(results.instance_scores.summary)
