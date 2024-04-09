from numpy import NaN
from unitxt import add_to_catalog
from unitxt.metrics import (
    PrecisionMacroMultiLabel,
    PrecisionMicroMultiLabel,
    RecallMacroMultiLabel,
    RecallMicroMultiLabel,
)
from unitxt.test_utils.metrics import test_metric

precision_micro_metric = PrecisionMicroMultiLabel()
precision_macro_metric = PrecisionMacroMultiLabel()
recall_micro_metric = RecallMicroMultiLabel()
recall_macro_metric = RecallMacroMultiLabel()


# Binary case: micro = macro
predictions = [["yes"], ["yes"], [], []]
references = [[["yes"]], [[]], [["yes"]], [["yes"]]]


instance_targets_precision_micro = [
    {"precision_micro": 1.0, "score": 1.0, "score_name": "precision_micro"},
    {"precision_micro": NaN, "score": NaN, "score_name": "precision_micro"},
    {"precision_micro": 0.0, "score": 0.0, "score_name": "precision_micro"},
    {"precision_micro": 0.0, "score": 0.0, "score_name": "precision_micro"},
]

global_target_precision_micro = {
    "precision_micro": 0.5,
    "score": 0.5,
    "score_name": "precision_micro",
    "score_ci_low": 0.5,
    "score_ci_high": 0.86,
    "precision_micro_ci_low": 0.5,
    "precision_micro_ci_high": 0.86,
}

instance_targets_precision_macro = [
    {"precision_macro": 1.0, "score": 1.0, "score_name": "precision_macro"},
    {"precision_macro": NaN, "score": NaN, "score_name": "precision_macro"},
    {"precision_macro": 0.0, "score": 0.0, "score_name": "precision_macro"},
    {"precision_macro": 0.0, "score": 0.0, "score_name": "precision_macro"},
]

global_target_precision_macro = {
    "precision_macro": 0.5,
    "score": 0.5,
    "score_name": "precision_macro",
    "score_ci_low": 0.5,
    "score_ci_high": 0.86,
    "precision_macro_ci_low": 0.5,
    "precision_macro_ci_high": 0.86,
}
outputs = test_metric(
    metric=precision_micro_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_precision_micro,
    global_target=global_target_precision_micro,
)
outputs = test_metric(
    metric=precision_macro_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_precision_macro,
    global_target=global_target_precision_macro,
)


global_target_recall_micro = {
    "recall_micro": 0.33,
    "score": 0.33,
    "score_name": "recall_micro",
    "score_ci_low": 0.33,
    "score_ci_high": 0.47,
    "recall_micro_ci_low": 0.33,
    "recall_micro_ci_high": 0.47,
}

instance_targets_recall_micro = [
    {"recall_micro": 1.0, "score": 1.0, "score_name": "recall_micro"},
    {"recall_micro": NaN, "score": NaN, "score_name": "recall_micro"},
    {"recall_micro": 0.0, "score": 0.0, "score_name": "recall_micro"},
    {"recall_micro": 0.0, "score": 0.0, "score_name": "recall_micro"},
]

global_target_recall_macro = {
    "recall_macro": 0.33,
    "score": 0.33,
    "score_name": "recall_macro",
    "score_ci_low": 0.33,
    "score_ci_high": 0.47,
    "recall_macro_ci_low": 0.33,
    "recall_macro_ci_high": 0.47,
}

instance_targets_recall_macro = [
    {"recall_macro": 1.0, "score": 1.0, "score_name": "recall_macro"},
    {"recall_macro": NaN, "score": NaN, "score_name": "recall_macro"},
    {"recall_macro": 0.0, "score": 0.0, "score_name": "recall_macro"},
    {"recall_macro": 0.0, "score": 0.0, "score_name": "recall_macro"},
]

outputs = test_metric(
    metric=recall_micro_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_recall_micro,
    global_target=global_target_recall_micro,
)
outputs = test_metric(
    metric=recall_macro_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_recall_macro,
    global_target=global_target_recall_macro,
)


# multi-class case

predictions = [["yes"], ["yes"], [], [], ["maybe"], ["maybe"], ["maybe"]]
references = [
    [["yes"]],
    [[]],
    [["yes"]],
    [["yes"]],
    [[]],
    [["maybe"]],
    [["yes"]],
]

instance_targets_precision_micro = [
    {"precision_micro": 1.0, "score": 1.0, "score_name": "precision_micro"},
    {"precision_micro": NaN, "score": NaN, "score_name": "precision_micro"},
    {"precision_micro": 0.0, "score": 0.0, "score_name": "precision_micro"},
    {"precision_micro": 0.0, "score": 0.0, "score_name": "precision_micro"},
    {"precision_micro": NaN, "score": NaN, "score_name": "precision_micro"},
    {"precision_micro": 1.0, "score": 1.0, "score_name": "precision_micro"},
    {"precision_micro": 0.0, "score": 0.0, "score_name": "precision_micro"},
]

instance_targets_precision_macro = [
    {"precision_macro": 1.0, "score": 1.0, "score_name": "precision_macro"},
    {"precision_macro": NaN, "score": NaN, "score_name": "precision_macro"},
    {"precision_macro": 0.0, "score": 0.0, "score_name": "precision_macro"},
    {"precision_macro": 0.0, "score": 0.0, "score_name": "precision_macro"},
    {"precision_macro": NaN, "score": NaN, "score_name": "precision_macro"},
    {"precision_macro": 1.0, "score": 1.0, "score_name": "precision_macro"},
    {"precision_macro": 0.0, "score": 0.0, "score_name": "precision_macro"},
]

global_target_precision_micro = {
    "precision_micro": 0.4,
    "score": 0.4,
    "score_name": "precision_micro",
    "score_ci_low": NaN,
    "score_ci_high": NaN,
    "precision_micro_ci_low": NaN,
    "precision_micro_ci_high": NaN,
}

global_target_precision_macro = {
    "precision_macro": 0.42,
    "score": 0.42,
    "score_name": "precision_macro",
    "score_ci_low": NaN,
    "score_ci_high": NaN,
    "precision_macro_ci_low": NaN,
    "precision_macro_ci_high": NaN,
}

outputs = test_metric(
    metric=precision_micro_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_precision_micro,
    global_target=global_target_precision_micro,
)

outputs = test_metric(
    metric=precision_macro_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_precision_macro,
    global_target=global_target_precision_macro,
)


global_target_recall_micro = {
    "recall_micro": 0.4,
    "score": 0.4,
    "score_name": "recall_micro",
    "score_ci_low": NaN,
    "score_ci_high": NaN,
    "recall_micro_ci_low": NaN,
    "recall_micro_ci_high": NaN,
}

global_target_recall_macro = {
    "recall_macro": 0.62,
    "score": 0.62,
    "score_name": "recall_macro",
    "score_ci_low": 0.63,
    "score_ci_high": 0.67,
    "recall_macro_ci_low": 0.63,
    "recall_macro_ci_high": 0.67,
}

instance_targets_recall_micro = [
    {
        "recall_micro": 1.0,
        "score": 1.0,
        "score_name": "recall_micro",
    },
    {
        "recall_micro": NaN,
        "score": NaN,
        "score_name": "recall_micro",
    },
    {
        "recall_micro": 0.0,
        "score": 0.0,
        "score_name": "recall_micro",
    },
    {
        "recall_micro": 0.0,
        "score": 0.0,
        "score_name": "recall_micro",
    },
    {
        "recall_micro": NaN,
        "score": NaN,
        "score_name": "recall_micro",
    },
    {
        "recall_micro": 1.0,
        "score": 1.0,
        "score_name": "recall_micro",
    },
    {
        "recall_micro": 0.0,
        "score": 0.0,
        "score_name": "recall_micro",
    },
]

instance_targets_recall_macro = [
    {
        "recall_macro": 1.0,
        "score": 1.0,
        "score_name": "recall_macro",
    },
    {
        "recall_macro": NaN,
        "score": NaN,
        "score_name": "recall_macro",
    },
    {
        "recall_macro": 0.0,
        "score": 0.0,
        "score_name": "recall_macro",
    },
    {
        "recall_macro": 0.0,
        "score": 0.0,
        "score_name": "recall_macro",
    },
    {
        "recall_macro": NaN,
        "score": NaN,
        "score_name": "recall_macro",
    },
    {
        "recall_macro": 1.0,
        "score": 1.0,
        "score_name": "recall_macro",
    },
    {
        "recall_macro": 0.0,
        "score": 0.0,
        "score_name": "recall_macro",
    },
]

outputs = test_metric(
    metric=recall_micro_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_recall_micro,
    global_target=global_target_recall_micro,
)

outputs = test_metric(
    metric=recall_macro_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_recall_macro,
    global_target=global_target_recall_macro,
)


add_to_catalog(recall_micro_metric, "metrics.recall_micro_multi_label", overwrite=True)
add_to_catalog(recall_macro_metric, "metrics.recall_macro_multi_label", overwrite=True)
add_to_catalog(
    precision_micro_metric, "metrics.precision_micro_multi_label", overwrite=True
)
add_to_catalog(
    precision_macro_metric, "metrics.precision_macro_multi_label", overwrite=True
)
