# This metric first computes query perplexity per instance, then
# computes recall at different topK vs reference relevance.

from unitxt import add_to_catalog
from unitxt.metrics import RerankRecall
from unitxt.test_utils.metrics import test_metric

metric = RerankRecall()

predictions = ["0.3", "0.9", "0.3", "0", "0.97", "bad_input"]
references = [["0"], ["1"], ["0"], ["0"], ["1"], ["0"]]
inputs = [
    {
        "query": "who is Barack Obama?",
        "query_id": "1",
        "passage": "Cats are animals",
        "passage_id": "1",
    },
    {
        "query": "who is Barack Obama?",
        "query_id": "1",
        "passage": "Obama was the president of the USA",
        "passage_id": "2",
    },
    {
        "query": "What are dogs?",
        "query_id": "2",
        "passage": "Cats are animals",
        "passage_id": "1",
    },
    {
        "query": "What are dogs?",
        "query_id": "2",
        "passage": "Obama was the president of the USA",
        "passage_id": "2",
    },
    {
        "query": "What are cats?",
        "query_id": "3",
        "passage": "Cats are animals",
        "passage_id": "1",
    },
    {
        "query": "What are cats?",
        "query_id": "3",
        "passage": "Obama was the president of the USA",
        "passage_id": "2",
    },
]
instance_targets = [  # nDCG is undefined at instance level
    {
        "recall_at_1": 0.0,
        "recall_at_2": 0.0,
        "recall_at_5": 0.0,
        "score": 0.0,
        "score_name": "recall_at_5",
    },
    {
        "recall_at_1": 1.0,
        "recall_at_2": 1.0,
        "recall_at_5": 1.0,
        "score": 1.0,
        "score_name": "recall_at_5",
    },
    {
        "recall_at_1": 0.0,
        "recall_at_2": 0.0,
        "recall_at_5": 0.0,
        "score": 0.0,
        "score_name": "recall_at_5",
    },
    {
        "recall_at_1": 0.0,
        "recall_at_2": 0.0,
        "recall_at_5": 0.0,
        "score": 0.0,
        "score_name": "recall_at_5",
    },
    {
        "recall_at_1": 1.0,
        "recall_at_2": 1.0,
        "recall_at_5": 1.0,
        "score": 1.0,
        "score_name": "recall_at_5",
    },
    {
        "recall_at_1": 0.0,
        "recall_at_2": 0.0,
        "recall_at_5": 0.0,
        "score": 0.0,
        "score_name": "recall_at_5",
    },
]

global_target = {
    "recall_at_1": 0.33,
    "recall_at_2": 0.67,
    "recall_at_5": 0.67,
    "score": 0.67,
    "score_name": "recall_at_5",
    "num_of_evaluated_instances": 6,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    task_data=inputs,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.rerank_recall", overwrite=True)
