from copy import deepcopy

from src.unitxt import add_to_catalog
from src.unitxt.metrics import (
    FixedGroupMeanAccuracy,
    GroupMeanAccuracy,
    GroupMeanStringContainment,
    GroupMeanTokenOverlap,
    GroupNormCohensHAccuracy,
    GroupNormCohensHStringContainment,
    GroupPDRAccuracy,
    GroupPDRStringContainment,
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

# possibly multi-column group identifier; 'ignore' is unused
# use deepcopy so that dicts in list are independent and can be updated separately
additional_inputs = (
    [deepcopy({"group": "grp1", "id": 0, "ignore": 1}) for _ in range(5)]
    + [deepcopy({"group": "grp1", "id": 1, "ignore": 1}) for _ in range(5)]
    + [deepcopy({"group": "grp2", "id": 0, "ignore": 1}) for _ in range(4)]
    + [deepcopy({"group": "grp2", "id": 1, "ignore": 0}) for _ in range(1)]
)
# for group_mean_subgroup_comparison metrics, add a subgroup indicator (by default called 'variant_type')
# these groupings correspond in length to the group identifiers above
variant_type = (
    (["original"] + ["paraphrase"] * 4)
    + (["original"] + ["paraphrase"] * 4)
    + (["original"] + ["paraphrase"] * 3)
    + ["original"]
)

# construct grouping_field by combining two other fields (and ignoring one); mimics what you would do in cards
group_by_fields = ["group", "id"]

for ai, vt in zip(additional_inputs, variant_type):
    ai.update(
        {
            "group_id": "_".join([str(ai[ff]) for ff in group_by_fields]),
            "variant_type": vt,
        }
    )


instance_targets_string_containment = [
    {"score": 1.0},
    {"score": 1.0},
    {"score": 0.0},
    {"score": 1.0},
    {"score": 0.0},
    {"score": 1.0},
    {"score": 1.0},
    {"score": 0.0},
    {"score": 0.0},
    {"score": 1.0},
    {"score": 1.0},
    {"score": 1.0},
    {"score": 1.0},
    {"score": 0.0},
    {"score": 0.0},
]

for instance in instance_targets_string_containment:
    instance.update(
        {"string_containment": instance["score"], "score_name": "string_containment"}
    )

instance_targets_accuracy = [
    {"score": 0.0},
    {"score": 1.0},
    {"score": 0.0},
    {"score": 0.0},
    {"score": 0.0},
    {"score": 1.0},
    {"score": 0.0},
    {"score": 0.0},
    {"score": 0.0},
    {"score": 0.0},
    {"score": 1.0},
    {"score": 1.0},
    {"score": 0.0},
    {"score": 0.0},
    {"score": 0.0},
]
for instance in instance_targets_accuracy:
    instance.update({"accuracy": instance["score"], "score_name": "accuracy"})

metric = FixedGroupMeanAccuracy()
global_target = {
    "group_fixed_group_mean_accuracy": 0.22,
    "score": 0.22,
    "score_name": "group_fixed_group_mean_accuracy",
    "score_ci_low": 0.1,
    "score_ci_high": 0.48,
    "group_fixed_group_mean_accuracy_ci_low": 0.1,
    "group_fixed_group_mean_accuracy_ci_high": 0.48,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_fixed_group_mean_accuracy", overwrite=True)


metric = GroupMeanAccuracy()
global_target = {
    "group_mean_accuracy": 0.22,
    "score": 0.22,
    "score_name": "group_mean_accuracy",
    "score_ci_low": 0.02,
    "score_ci_high": 0.44,
    "group_mean_accuracy_ci_low": 0.02,
    "group_mean_accuracy_ci_high": 0.44,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_mean_accuracy", overwrite=True)


metric = GroupMeanStringContainment()
global_target = {
    "group_mean_string_containment": 0.49,
    "score": 0.49,
    "score_name": "group_mean_string_containment",
    "score_ci_low": 0.16,
    "score_ci_high": 0.71,
    "group_mean_string_containment_ci_low": 0.16,
    "group_mean_string_containment_ci_high": 0.71,
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
metric = GroupPDRAccuracy()
global_target = {
    "group_pdr_accuracy": 0.83,
    "score": 0.83,
    "score_name": "group_pdr_accuracy",
    "score_ci_low": 0.67,
    "score_ci_high": 1.0,
    "group_pdr_accuracy_ci_low": 0.67,
    "group_pdr_accuracy_ci_high": 1.0,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_pdr_accuracy", overwrite=True)


metric = GroupPDRStringContainment()
global_target = {
    "group_pdr_string_containment": 0.44,
    "score": 0.44,
    "score_name": "group_pdr_string_containment",
    "score_ci_low": 0.33,
    "score_ci_high": 0.5,
    "group_pdr_string_containment_ci_low": 0.33,
    "group_pdr_string_containment_ci_high": 0.5,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_pdr_string_containment", overwrite=True)

# Cohen's H
metric = GroupNormCohensHAccuracy()
global_target = {
    "group_norm_cohens_h_accuracy": -0.42,
    "score": -0.42,
    "score_name": "group_norm_cohens_h_accuracy",
    "score_ci_low": -1.0,
    "score_ci_high": 0.33,
    "group_norm_cohens_h_accuracy_ci_low": -1.0,
    "group_norm_cohens_h_accuracy_ci_high": 0.33,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_norm_cohens_h_accuracy", overwrite=True)


metric = GroupNormCohensHStringContainment()
global_target = {
    "group_norm_cohens_h_string_containment": -0.46,
    "score": -0.46,
    "score_name": "group_norm_cohens_h_string_containment",
    "score_ci_low": -0.5,
    "score_ci_high": -0.39,
    "group_norm_cohens_h_string_containment_ci_low": -0.5,
    "group_norm_cohens_h_string_containment_ci_high": -0.39,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_norm_cohens_h_string_containment", overwrite=True)

global_target = {
    "group_mean_f1": 0.51,
    "score": 0.51,
    "score_name": "group_mean_f1",
    "group_mean_f1_ci_low": 0.22,
    "group_mean_f1_ci_high": 0.68,
    "score_ci_low": 0.22,
    "score_ci_high": 0.68,
    "group_mean_precision": 0.5,
    "group_mean_precision_ci_low": 0.21,
    "group_mean_precision_ci_high": 0.67,
    "group_mean_recall": 0.52,
    "group_mean_recall_ci_low": 0.25,
    "group_mean_recall_ci_high": 0.71,
}

instance_targets_token_overlap = [
    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0, "score_name": "f1"},
    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0, "score_name": "f1"},
    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0, "score_name": "f1"},
    {"precision": 0, "recall": 0, "f1": 0, "score": 0, "score_name": "f1"},
    {"precision": 0, "recall": 0, "f1": 0, "score": 0, "score_name": "f1"},
    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0, "score_name": "f1"},
    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0, "score_name": "f1"},
    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0, "score_name": "f1"},
    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0, "score_name": "f1"},
    {"precision": 0.5, "recall": 1.0, "f1": 0.67, "score": 0.67, "score_name": "f1"},
    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0, "score_name": "f1"},
    {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0, "score_name": "f1"},
    {"precision": 0, "recall": 0, "f1": 0, "score": 0, "score_name": "f1"},
    {"precision": 0, "recall": 0, "f1": 0, "score": 0, "score_name": "f1"},
    {"precision": 0, "recall": 0, "f1": 0, "score": 0, "score_name": "f1"},
]


metric = GroupMeanTokenOverlap()

outputs = test_metric(
    metric=metric,
    predictions=[str(vv) for vv in predictions],
    references=[[str(vvv) for vvv in vv] for vv in references],
    instance_targets=instance_targets_token_overlap,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.group_mean_token_overlap", overwrite=True)
