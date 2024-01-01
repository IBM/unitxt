import numpy as np

from src.unitxt import add_to_catalog
from src.unitxt.metrics import (
    MeanGroupedAccuracy,
    MeanGroupedAccuracyPDR,
    MeanGroupedF1MacroMultiLabel,
    MeanGroupedStringContainment,
    MeanGroupedStringContainmentPDR,
)
from src.unitxt.test_utils.metrics import test_metric

predictions = [
    "A B",
    "BC D",
    "C",
    "123",
    "BCD",
    10,
    "  BD",
    "AB",
    "I am a dog",
    "AB C",
    "AB 1",
    "GMA",
    0.123,
    "BD",
    "abc",
]

references = [
    ["B", "AB", "A"],
    ["A", "BC D", "BC DF"],
    ["c", " C"],
    [13, 23, 234],
    ["  ", " BD", " BDA"],
    [1, 10, 100],
    ["A", "B", "BD"],
    ["ABC", "ab", "BC"],
    ["I am a person", "I AM A DOG", "ABC"],
    ["AB CD", "AB", "ab"],
    ["AB 1", "AB1"],
    [" GMA 123", "GMA"],
    ["123", 0.12],
    ["BDE", "BCE", "bdefs"],
    [" abcdefg", "AB", "abcd"],
]

# possibly multi-column group identifier
additional_inputs = (
    [{"group": "grp1", "id": 0, "ignore": 1}] * 5
    + [{"group": "grp1", "id": 1, "ignore": 1}] * 5
    + [{"group": "grp2", "id": 0, "ignore": 1}] * 4
    + [{"group": "grp2", "id": 1, "ignore": 0}] * 1
)

group_by_fields = ["group", "id"]
# construct grouping_field by combining two other fields (and ignoring one); mimics what you would do in cards
for ai in additional_inputs:
    ai.update({"group_id": "_".join([str(ai[ff]) for ff in group_by_fields])})


instance_targets_string_containment = [
    {
        "group_mean_string_containment": 1.0,
        "score": 1.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 1.0,
        "score": 1.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 0.0,
        "score": 0.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 1.0,
        "score": 1.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 0.0,
        "score": 0.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 1.0,
        "score": 1.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 1.0,
        "score": 1.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 0.0,
        "score": 0.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 0.0,
        "score": 0.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 1.0,
        "score": 1.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 1.0,
        "score": 1.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 1.0,
        "score": 1.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 1.0,
        "score": 1.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 0.0,
        "score": 0.0,
        "score_name": "group_mean_string_containment",
    },
    {
        "group_mean_string_containment": 0.0,
        "score": 0.0,
        "score_name": "group_mean_string_containment",
    },
]

instance_targets_exact = [
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 1.0, "score": 1.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 1.0, "score": 1.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 1.0, "score": 1.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 1.0, "score": 1.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
    {"group_mean_accuracy": 0.0, "score": 0.0, "score_name": "group_mean_accuracy"},
]

# for PDR, metric is undefined on a single instance
instance_targets_exact_pdr = [
    {
        "group_mean_accuracy_pdr": np.nan,
        "score": np.nan,
        "score_name": "group_mean_accuracy_pdr",
    }
] * len(references)

instance_targets_string_containment_pdr = [
    {
        "group_mean_string_containment_pdr": np.nan,
        "score": np.nan,
        "score_name": "group_mean_string_containment_pdr",
    }
] * len(references)


metric = MeanGroupedAccuracy()
global_target = {
    "group_mean_accuracy": 0.23,
    "score": 0.23,
    "score_name": "group_mean_accuracy",
    "score_ci_low": 0.0,
    "score_ci_high": 0.45,
    "group_mean_accuracy_ci_low": 0.0,
    "group_mean_accuracy_ci_high": 0.45,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_exact,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_mean_accuracy", overwrite=True)


metric = MeanGroupedStringContainment()
global_target = {
    "group_mean_string_containment": 0.49,
    "score": 0.49,
    "score_name": "group_mean_string_containment",
    "score_ci_low": 0.17,
    "score_ci_high": 0.73,
    "group_mean_string_containment_ci_low": 0.17,
    "group_mean_string_containment_ci_high": 0.73,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_mean_string_containment", overwrite=True)


# PDR
metric = MeanGroupedAccuracyPDR()
global_target = {
    "group_mean_accuracy_pdr": 0.83,
    "score": 0.83,
    "score_name": "group_mean_accuracy_pdr",
    "score_ci_low": 0.5,
    "score_ci_high": 1.0,
    "group_mean_accuracy_pdr_ci_low": 0.5,
    "group_mean_accuracy_pdr_ci_high": 1.0,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_exact_pdr,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_mean_accuracy_pdr", overwrite=True)


metric = MeanGroupedStringContainmentPDR()
global_target = {
    "group_mean_string_containment_pdr": 0.44,
    "score": 0.44,
    "score_name": "group_mean_string_containment_pdr",
    "score_ci_low": 0.12,
    "score_ci_high": 1.0,
    "group_mean_string_containment_pdr_ci_low": 0.12,
    "group_mean_string_containment_pdr_ci_high": 1.0,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment_pdr,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_mean_string_containment_pdr", overwrite=True)


# F1 requires different predictions and references
f1_predictions = [
    "A",
    "B",
    "B",
    "A",
    "B",
    "B",
    "A",
    "A",
    "B",
    "B",
    "A",
    "B",
    "A",
    "A",
    "B",
]
f1_predictions = [[pp] for pp in f1_predictions]

f1_references = [
    ["A", "B"],
    ["A", "C"],
    ["B", "C", "A"],
    ["A"],
    ["B", "A"],
    ["C", "B"],
    ["A"],
    ["B", "C"],
    ["A", "B", "C"],
    ["A", "B"],
    ["B", "C"],
    ["C"],
    ["C", "B"],
    ["B", "A"],
    ["B"],
]
f1_references = [[rr] for rr in f1_references]
instance_targets_f1 = [
    {"group_mean_f1_macro": 0.5, "score": 0.5, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.0, "score": 0.0, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.33, "score": 0.33, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 1.0, "score": 1.0, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.5, "score": 0.5, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.5, "score": 0.5, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 1.0, "score": 1.0, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.0, "score": 0.0, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.33, "score": 0.33, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.5, "score": 0.5, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.0, "score": 0.0, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.0, "score": 0.0, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.0, "score": 0.0, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 0.5, "score": 0.5, "score_name": "group_mean_f1_macro"},
    {"group_mean_f1_macro": 1.0, "score": 1.0, "score_name": "group_mean_f1_macro"},
]


global_target = {
    "group_mean_f1_macro": 0.51,
    "group_mean_f1_macro_ci_high": 0.73,
    "group_mean_f1_macro_ci_low": 0.39,
    "score": 0.51,
    "score_ci_high": 0.73,
    "score_ci_low": 0.39,
    "score_name": "group_mean_f1_macro",
}
metric = MeanGroupedF1MacroMultiLabel()

outputs = test_metric(
    metric=metric,
    predictions=f1_predictions,
    references=f1_references,
    instance_targets=instance_targets_f1,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_mean_f1_macro_multilabel", overwrite=True)
