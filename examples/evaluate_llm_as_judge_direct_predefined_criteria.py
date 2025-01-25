from unitxt import get_logger
from unitxt.api import create_dataset, evaluate

logger = get_logger()

data = [
    {"question": "Who is Harry Potter?"},
    {"question": "How can I protect myself from the wind while walking outside?"},
    {"question": "What is a good low cost of living city in the US?"},
]

criterion = "metrics.llm_as_judge.direct.criterias.answer_relevance"
metrics = [
    f"metrics.llm_as_judge.direct.rits.llama3_1_70b[criteria={criterion}, context_fields=[question]]"
]

dataset = create_dataset(
    task="tasks.qa.open", test_set=data, metrics=metrics, split="test"
)

predictions = [
    """Harry Potter is a young wizard who becomes famous for surviving an attack by the dark wizard Voldemort, and later embarks on a journey to defeat him and uncover the truth about his past.""",
    """You can protect yourself from the wind by wearing windproof clothing, layering up, and using accessories like hats, scarves, and gloves to cover exposed skin.""",
    """A good low-cost-of-living city in the U.S. is San Francisco, California, known for its affordable housing and budget-friendly lifestyle.""",
]

results = evaluate(predictions=predictions, data=dataset)

print("Global Scores:")
print(results.global_scores.summary)

print("Instance Scores:")
print(results.instance_scores)
