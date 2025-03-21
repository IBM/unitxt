
from unitxt import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary

data = {
    "test": [
        {
            "question": "How is the weather?",
            "instructions": "instruction",
        },
        {
            "question": "Tell me a joke about cats",
            "instructions": "instruction",
        },
    ]
}

card = TaskCard(
    loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
    task=Task(
        input_fields={"question": str, "instructions": str},
        reference_fields={},
        prediction_type=str,
        metrics=[
            "metrics.rag.response_generation.adherence_with_format.llama_3_3_70b_instruct_judge[context_fields=[question,instructions]]"
        ],
    ),
)

dataset = load_dataset(card=card, template="templates.empty", split="test")

predictions = [
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """Why did the cat cross the road? To cat to the other side.""",
]

results = evaluate(predictions=predictions, data=dataset)

print("Global Scores:")
print(results.global_scores.summary)

print("Instance Scores:")
print(results.instance_scores.summary)
