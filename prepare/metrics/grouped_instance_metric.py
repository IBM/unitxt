from src.unitxt import add_to_catalog
from src.unitxt.metrics import MeanGroupedAccuracy#, MeanGroupedSubstringAccuracy, MeanGroupedAccuracyPerformanceDrop, MeanGroupedSubstringAccuracyPerformanceDrop
from src.unitxt.test_utils.metrics import test_metric


predictions = ["A B", "BC D", "C", "123", "BCD",
               10, "  BD", "AB", "I am a dog", "AB C",
              "AB 1", "GMA", 0.123, "BD", "abc"]

references = [["B", "AB", "A"], ["A", "BC D", "BC DF"], ["c", " C"], [13, 23, 234], ["  ", " BD", " BDA"],
               [1, 10, 100], ["A", "B", "BD"], ["ABC", "ab", "BC"], ["I am a person", "I AM A DOG", "ABC"], ["AB CD", "AB", "ab"],
               ["AB 1", "AB1"], [" GMA 123", "GMA"], ["123", 0.12], ["BDE", "BCE", "bdefs"], [' abcdefg', 'AB', 'abcd']]

# possibly multi-column group identifier
additional_inputs = [{"group": "grp1", "id": 0}] * 5 + [{"group": "grp1", "id": 1}] * 5 + [{"group": "grp2", "id": 0}] * 4 + [{"group": "grp2", "id": 1}] * 1

substring_instance_targets = [
    {"substring_accuracy": 0.0, "score": 0.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 1.0, "score": 1.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 1.0, "score": 1.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 0.0, "score": 0.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 0.0, "score": 0.0, "score_name": "substring_accuracy"},

    {"substring_accuracy": 1.0, "score": 1.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 0.0, "score": 0.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 1.0, "score": 1.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 0.0, "score": 0.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 1.0, "score": 1.0, "score_name": "substring_accuracy"},

    {"substring_accuracy": 1.0, "score": 1.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 1.0, "score": 1.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 0.0, "score": 0.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 1.0, "score": 1.0, "score_name": "substring_accuracy"},
    {"substring_accuracy": 1.0, "score": 1.0, "score_name": "substring_accuracy"},
]

exact_instance_targets = [
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},

    {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},

    {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
    {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
    {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
]


metric = MeanGroupedAccuracy()
global_target = {"accuracy": 0.225, "score": 0.225, "score_name": "accuracy"}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=exact_instance_targets,
    global_target=global_target,
    additional_inputs=additional_inputs
)

add_to_catalog(metric, "metrics.mean_grouped_accuracy", overwrite=True)

#
# metric = MeanGroupedSubstringAccuracy()
# global_target = {"accuracy": 0.5, "score": 0.5, "score_name": "accuracy"}
#
# outputs = test_metric(
#     metric=metric,
#     predictions=predictions,
#     references=references,
#     instance_targets=substring_instance_targets,
#     global_target=global_target,
#     additional_inputs=additional_inputs
# )
#
# metric = MeanGroupedAccuracyPerformanceDrop()
# global_target = {"accuracy": 0.5, "score": 0.5, "score_name": "accuracy"}
#
# outputs = test_metric(
#     metric=metric,
#     predictions=predictions,
#     references=references,
#     instance_targets=exact_instance_targets,
#     global_target=global_target,
#     additional_inputs=additional_inputs
# )
#
# metric = MeanGroupedSubstringAccuracyPerformanceDrop()
# global_target = {"accuracy": 0.5, "score": 0.5, "score_name": "accuracy"}
#
# outputs = test_metric(
#     metric=metric,
#     predictions=predictions,
#     references=references,
#     instance_targets=substring_instance_targets,
#     global_target=global_target,
#     additional_inputs=additional_inputs
# )

