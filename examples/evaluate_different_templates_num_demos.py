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
    demos_pool_size=100,
    loader_limit=200,
)

test = dataset["test"].to_list()
predictions = ["entailment" for _ in test]

results = evaluate(predictions=predictions, data=test)

# Print the resulting scores per group.
logger.info(results[0]["score"]["groups"])
