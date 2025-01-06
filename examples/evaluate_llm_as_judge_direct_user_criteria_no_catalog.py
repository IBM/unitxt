from unitxt.api import create_dataset, evaluate
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.llm_as_judge import LLMJudgeDirect
from unitxt.llm_as_judge_constants import (
    CriteriaWithOptions,
)

criteria = CriteriaWithOptions.from_obj(
    {
        "name": "Temperature in Fahrenheit and Celsius",
        "description": "In the response, if there is a numerical temperature present, is it denominated in both Fahrenheit and Celsius?",
        "options": [
            {
                "name": "Yes",
                "description": "The temperature reading is provided in both Fahrenheit and Celsius.",
            },
            {
                "name": "No",
                "description": "The temperature reading is provided either in Fahrenheit or Celsius, but not both.",
            },
            {
                "name": "Pass",
                "description": "There is no numerical temperature reading in the response.",
            },
        ],
        "option_map": {"Yes": 1.0, "No": 0.5, "Pass": 0.0},
    }
)


data = [
    {"question": "How is the weather?"},
    {"question": "How is the weather?"},
    {"question": "How is the weather?"},
]

metric = LLMJudgeDirect(
    inference_engine=CrossProviderInferenceEngine(
        model="llama-3-1-70b-instruct", max_tokens=1024
    ),
    criteria=criteria,
    context_fields=["question"],
    criteria_field="criteria",
)

dataset = create_dataset(
    task="tasks.qa.open", test_set=data, metrics=[metric], split="test"
)

predictions = [
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
]

results = evaluate(predictions=predictions, data=dataset)

print("Global Scores:")
print(results.global_scores.summary)

print("Instance Scores:")
print(results.instance_scores.summary)
