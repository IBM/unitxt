import numpy as np

from src.unitxt import add_to_catalog
from src.unitxt.metrics import (
    MeanGroupedAccuracy,
    MeanGroupedAccuracyPDR,
    MeanGroupedStringContainment,
    MeanGroupedStringContainmentPDR,
)
from src.unitxt.test_utils.metrics import test_metric

predictions = ["A B", "BC D", "C", "123", "BCD",
               10, "  BD", "AB", "I am a dog", "AB C",
              "AB 1", "GMA", 0.123, "BD", "abc"]

references = [["B", "AB", "A"], ["A", "BC D", "BC DF"], ["c", " C"], [13, 23, 234], ["  ", " BD", " BDA"],
               [1, 10, 100], ["A", "B", "BD"], ["ABC", "ab", "BC"], ["I am a person", "I AM A DOG", "ABC"], ["AB CD", "AB", "ab"],
               ["AB 1", "AB1"], [" GMA 123", "GMA"], ["123", 0.12], ["BDE", "BCE", "bdefs"], [" abcdefg", "AB", "abcd"]]

# possibly multi-column group identifier
additional_inputs = [{"group": "grp1", "id": 0}] * 5 + [{"group": "grp1", "id": 1}] * 5 + [{"group": "grp2", "id": 0}] * 4 + [{"group": "grp2", "id": 1}] * 1

instance_targets_string_containment = [
    {"string_containment": 1.0, "score": 1.0, "score_name": "string_containment"},
    {"string_containment": 1.0, "score": 1.0, "score_name": "string_containment"},
    {"string_containment": 0.0, "score": 0.0, "score_name": "string_containment"},
    {"string_containment": 1.0, "score": 1.0, "score_name": "string_containment"},
    {"string_containment": 0.0, "score": 0.0, "score_name": "string_containment"},

    {"string_containment": 1.0, "score": 1.0, "score_name": "string_containment"},
    {"string_containment": 1.0, "score": 1.0, "score_name": "string_containment"},
    {"string_containment": 0.0, "score": 0.0, "score_name": "string_containment"},
    {"string_containment": 0.0, "score": 0.0, "score_name": "string_containment"},
    {"string_containment": 1.0, "score": 1.0, "score_name": "string_containment"},

    {"string_containment": 1.0, "score": 1.0, "score_name": "string_containment"},
    {"string_containment": 1.0, "score": 1.0, "score_name": "string_containment"},
    {"string_containment": 1.0, "score": 1.0, "score_name": "string_containment"},
    {"string_containment": 0.0, "score": 0.0, "score_name": "string_containment"},
    {"string_containment": 0.0, "score": 0.0, "score_name": "string_containment"},
]

instance_targets_exact = [
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

# for PDR, metric is undefined on a single instance
instance_targets_exact_pdr = [{"accuracy_pdr": np.nan, "score": np.nan, "score_name": "accuracy_pdr"}] * len(references)
instance_targets_string_containment_pdr = [{"string_containment_pdr": np.nan, "score": np.nan, "score_name": "string_containment_pdr"}] * len(references)

metric = MeanGroupedAccuracy()
global_target = {"accuracy": 0.23, "score": 0.23,
                 "score_name": "accuracy",
                 "score_ci_low": np.nan, "score_ci_high": np.nan,
                 "accuracy_ci_low": np.nan, "accuracy_ci_high": np.nan}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_exact,
    global_target=global_target,
    additional_inputs=additional_inputs
)

add_to_catalog(metric, "metrics.mean_grouped_accuracy", overwrite=True)

metric = MeanGroupedStringContainment()
global_target = {"string_containment": 0.49, "score": 0.49,
                 "score_name": "string_containment",
                 "score_ci_low": np.nan, "score_ci_high": np.nan,
                 "string_containment_ci_low": np.nan, "string_containment_ci_high": np.nan}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    additional_inputs=additional_inputs
)

add_to_catalog(metric, "metrics.mean_grouped_string_containment", overwrite=True)



# PDR
metric = MeanGroupedAccuracyPDR()
global_target = {"accuracy_pdr": 0.83, "score": 0.83,
                 "score_name": "accuracy_pdr",
                 "score_ci_low": np.nan, "score_ci_high": np.nan,
                 "accuracy_pdr_ci_low": np.nan, "accuracy_pdr_ci_high": np.nan}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_exact_pdr,
    global_target=global_target,
    additional_inputs=additional_inputs
)

add_to_catalog(metric, "metrics.mean_grouped_accuracy_pdr", overwrite=True)


metric = MeanGroupedStringContainmentPDR()
global_target = {"string_containment_pdr": 0.44, "score": 0.44,
                 "score_name": "string_containment_pdr",
                 "score_ci_low": 0.19, "score_ci_high": 0.62,
                 "string_containment_pdr_ci_low": 0.19, "string_containment_pdr_ci_high": 0.62}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment_pdr,
    global_target=global_target,
    additional_inputs=additional_inputs
)

add_to_catalog(metric, "metrics.mean_grouped_string_containment_pdr", overwrite=True)
