from unitxt import add_to_catalog
from unitxt.metrics import JaccardIndex
from unitxt.test_utils.metrics import test_metric

metric = JaccardIndex()

predictions = [["A", "B", "C"]]
references = [[["B", "A", "D"]]]

instance_targets = [
    {"jaccard_index": 0.5, "score": 0.5, "score_name": "jaccard_index"},
]

global_target = {
    "jaccard_index": 0.5,
    "score": 0.5,
    "score_name": "jaccard_index",
    "num_of_evaluated_instances": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.jaccard_index", overwrite=True)
