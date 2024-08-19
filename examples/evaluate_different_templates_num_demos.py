import json

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
    demos_pool_size=100,
    loader_limit=200,
)

# Print the resulting dataset.
for num_demos in [0, 5]:
    for template in [
        "templates.classification.multi_class.relation.default",
        "templates.key_val",
    ]:
        subset = []
        for instance in dataset["test"]:
            metadata = json.loads(instance["task_data"])["metadata"]
            if metadata["num_demos"] == num_demos and metadata["template"] == template:
                subset.append(instance)

        # Generate predictions which are always entailment. Can be replaced with any inference method.
        predictions = ["entailment" for t in subset]

        evaluated_dataset = evaluate(predictions=predictions, data=subset)

        # Get the final score for that subset
        score = evaluated_dataset[0]["score"]["global"]["score"]

        logger.info(f"Num Demos: {num_demos}, Template: {template}, Score: {score}")
