from unitxt import evaluate, load_dataset
from unitxt.logging_utils import get_logger

logger = get_logger()

dataset = load_dataset(
    card="cards.wnli",
    template=[
        "templates.classification.multi_class.relation.default",
        "templates.key_val",
    ],
    num_demos=[0, 5],
    group_by=["template", "num_demos", ["template", "num_demos"]],
    demos_pool_size=10,
    loader_limit=200,
    max_test_instances=10,
    split="test",
)

predictions = ["entailment" for _ in dataset]

results = evaluate(predictions=predictions, data=dataset)

# Print Results:

print("Global Results:")
print(results.global_scores.summary)

print("Groups Results:")
print(results.groups_scores.summary)
