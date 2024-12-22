from unitxt import get_logger
from unitxt.api import create_dataset, evaluate

logger = get_logger()

data = [
    {"question": "How is the weather?"},
    {"question": "How is the weather?"},
    {"question": "How is the weather?"},
]

criteria = "metrics.llm_as_judge.direct.criterias.temperature_in_celsius_and_fahrenheit"
metrics = [
    f"metrics.llm_as_judge.direct.rits.llama3_1_70b[criteria={criteria}, context_fields=[question]]"
]

dataset = create_dataset(
    task="tasks.qa.open", test_set=data, metrics=metrics, split="test"
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
