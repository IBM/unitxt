from unitxt import add_to_catalog
from unitxt.metrics import JaccardIndex, JaccardIndexString
from unitxt.string_operators import RegexSplit
from unitxt.test_utils.metrics import test_metric

metric = JaccardIndex(
    __description__ = """JaccardIndex metric that operates on predictions and references that are list of elements.
    For each prediction, it calculates the score as Intersect(prediction,reference)/Union(prediction,reference).
    If multiple references exist, it takes for each predictions, the best ratio achieved by one of the references.
    It then aggregates the mean over all references.

    Note the metric assumes the prediction and references are either a set of elements or a list of elements.
    If the prediction and references are strings use JaccardIndexString metrics like "metrics.jaccard_index_words" .
    """)

predictions = [["A", "B", "C"]]
references = [[["B", "A", "D"]]]

instance_targets = [
    {"jaccard_index": 0.5, "score": 0.5, "score_name": "jaccard_index"},
]

global_target = {
    "jaccard_index": 0.5,
    "score": 0.5,
    "score_name": "jaccard_index",
    "num_of_instances": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.jaccard_index", overwrite=True)



metric = JaccardIndexString(  __description__ = """JaccardIndex metric that operates on prediction and references that are strings.
    It first splits the the string into words using space as a separator.

    For each prediction, it calculates the ratio Intersect(prediction_words,reference_words)/Union(prediction_words,reference_words).
    If multiple references exist, it takes the best ratio achieved by one of the references.

    """,
    splitter = RegexSplit(by=r"\s+")
    )

predictions = ["A B C"]
references = [["B  A  D"]]

instance_targets = [
    {"jaccard_index": 0.5, "score": 0.5, "score_name": "jaccard_index"},
]

global_target = {
    "jaccard_index": 0.5,
    "score": 0.5,
    "score_name": "jaccard_index",
    "num_of_instances": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.jaccard_index_words", overwrite=True)
