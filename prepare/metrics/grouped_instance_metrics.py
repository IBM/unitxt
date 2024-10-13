from unitxt import add_to_catalog
from unitxt.metrics import (
    FixedGroupAbsvalNormCohensHParaphraseAccuracy,
    FixedGroupAbsvalNormCohensHParaphraseStringContainment,
    FixedGroupAbsvalNormHedgesGParaphraseAccuracy,
    FixedGroupAbsvalNormHedgesGParaphraseStringContainment,
    FixedGroupMeanAccuracy,
    FixedGroupMeanBaselineAccuracy,
    FixedGroupMeanBaselineStringContainment,
    FixedGroupMeanParaphraseAccuracy,
    FixedGroupMeanParaphraseStringContainment,
    FixedGroupMeanStringContainment,
    FixedGroupNormCohensHParaphraseAccuracy,
    FixedGroupNormCohensHParaphraseStringContainment,
    FixedGroupNormHedgesGParaphraseAccuracy,
    FixedGroupNormHedgesGParaphraseStringContainment,
    FixedGroupPDRParaphraseAccuracy,
    FixedGroupPDRParaphraseStringContainment,
    GroupMeanAccuracy,
    GroupMeanStringContainment,
    GroupMeanTokenOverlap,
)
from unitxt.test_utils.metrics import test_metric

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

# task_data, consisting of a group_id (group instance scores by this, then apply aggregation function)
# and variant_type (for metrics that compare, say original vs paraphrase instance score)
# create 4 groups, of sizes 5,5,4,1
task_data = [
    {"group_id": "group1", "variant_type": "original"},
    {"group_id": "group1", "variant_type": "paraphrase"},
    {"group_id": "group1", "variant_type": "paraphrase"},
    {"group_id": "group1", "variant_type": "paraphrase"},
    {"group_id": "group1", "variant_type": "paraphrase"},
    {"group_id": "group2", "variant_type": "original"},
    {"group_id": "group2", "variant_type": "paraphrase"},
    {"group_id": "group2", "variant_type": "paraphrase"},
    {"group_id": "group2", "variant_type": "paraphrase"},
    {"group_id": "group2", "variant_type": "paraphrase"},
    {"group_id": "group3", "variant_type": "original"},
    {"group_id": "group3", "variant_type": "paraphrase"},
    {"group_id": "group3", "variant_type": "paraphrase"},
    {"group_id": "group3", "variant_type": "paraphrase"},
    {"group_id": "group4", "variant_type": "original"},
]


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

# now test the metrics
# group mean accuracy, fixed and not

metric = FixedGroupMeanAccuracy()
global_target = {
    "fixed_group_mean_accuracy": 0.22,
    "score": 0.22,
    "score_name": "fixed_group_mean_accuracy",
    "score_ci_low": 0.1,
    "score_ci_high": 0.48,
    "fixed_group_mean_accuracy_ci_low": 0.1,
    "fixed_group_mean_accuracy_ci_high": 0.48,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(metric, "metrics.robustness.fixed_group_mean_accuracy", overwrite=True)


metric = GroupMeanAccuracy()
global_target = {
    "group_mean_accuracy": 0.22,
    "score": 0.22,
    "score_name": "group_mean_accuracy",
    "score_ci_low": 0.02,
    "score_ci_high": 0.44,
    "group_mean_accuracy_ci_low": 0.02,
    "group_mean_accuracy_ci_high": 0.44,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(metric, "metrics.robustness.group_mean_accuracy", overwrite=True)

# group mean string containment, fixed and not

metric = FixedGroupMeanStringContainment()
global_target = {
    "fixed_group_mean_string_containment": 0.49,
    "score": 0.49,
    "score_name": "fixed_group_mean_string_containment",
    "score_ci_low": 0.0,
    "score_ci_high": 0.68,
    "fixed_group_mean_string_containment_ci_low": 0.0,
    "fixed_group_mean_string_containment_ci_high": 0.68,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric, "metrics.robustness.fixed_group_mean_string_containment", overwrite=True
)


metric = GroupMeanStringContainment()
global_target = {
    "group_mean_string_containment": 0.49,
    "score": 0.49,
    "score_name": "group_mean_string_containment",
    "score_ci_low": 0.16,
    "score_ci_high": 0.71,
    "group_mean_string_containment_ci_low": 0.16,
    "group_mean_string_containment_ci_high": 0.71,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric, "metrics.robustness.group_mean_string_containment", overwrite=True
)


# Group mean of baseline or other scores
metric = FixedGroupMeanBaselineAccuracy()
global_target = {
    "fixed_group_mean_baseline_accuracy": 0.5,
    "score": 0.5,
    "score_name": "fixed_group_mean_baseline_accuracy",
    "score_ci_low": 0.0,
    "score_ci_high": 1.0,
    "fixed_group_mean_baseline_accuracy_ci_low": 0.0,
    "fixed_group_mean_baseline_accuracy_ci_high": 1.0,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric, "metrics.robustness.fixed_group_mean_baseline_accuracy", overwrite=True
)

metric = FixedGroupMeanParaphraseAccuracy()
global_target = {
    "fixed_group_mean_paraphrase_accuracy": 0.19,
    "score": 0.19,
    "score_name": "fixed_group_mean_paraphrase_accuracy",
    "score_ci_low": 0.0,
    "score_ci_high": 0.33,
    "fixed_group_mean_paraphrase_accuracy_ci_low": 0.0,
    "fixed_group_mean_paraphrase_accuracy_ci_high": 0.33,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric, "metrics.robustness.fixed_group_mean_paraphrase_accuracy", overwrite=True
)


metric = FixedGroupMeanBaselineStringContainment()
global_target = {
    "fixed_group_mean_baseline_string_containment": 0.75,
    "score": 0.75,
    "score_name": "fixed_group_mean_baseline_string_containment",
    "score_ci_low": 0.25,
    "score_ci_high": 1.0,
    "fixed_group_mean_baseline_string_containment_ci_low": 0.25,
    "fixed_group_mean_baseline_string_containment_ci_high": 1.0,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_mean_baseline_string_containment",
    overwrite=True,
)

metric = FixedGroupMeanParaphraseStringContainment()
global_target = {
    "fixed_group_mean_paraphrase_string_containment": 0.56,
    "score": 0.56,
    "score_name": "fixed_group_mean_paraphrase_string_containment",
    "score_ci_low": 0.5,
    "score_ci_high": 0.67,
    "fixed_group_mean_paraphrase_string_containment_ci_low": 0.5,
    "fixed_group_mean_paraphrase_string_containment_ci_high": 0.67,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_mean_paraphrase_string_containment",
    overwrite=True,
)


# PDR: will always use fixed groups
metric = FixedGroupPDRParaphraseAccuracy()
global_target = {
    "fixed_group_pdr_paraphrase_accuracy": 0.83,
    "score": 0.83,
    "score_name": "fixed_group_pdr_paraphrase_accuracy",
    "score_ci_low": 0.67,
    "score_ci_high": 1.0,
    "fixed_group_pdr_paraphrase_accuracy_ci_low": 0.67,
    "fixed_group_pdr_paraphrase_accuracy_ci_high": 1.0,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric, "metrics.robustness.fixed_group_pdr_paraphrase_accuracy", overwrite=True
)

metric = FixedGroupPDRParaphraseStringContainment()
global_target = {
    "fixed_group_pdr_paraphrase_string_containment": 0.44,
    "score": 0.44,
    "score_name": "fixed_group_pdr_paraphrase_string_containment",
    "score_ci_low": 0.33,
    "score_ci_high": 0.5,
    "fixed_group_pdr_paraphrase_string_containment_ci_low": 0.33,
    "fixed_group_pdr_paraphrase_string_containment_ci_high": 0.5,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_pdr_paraphrase_string_containment",
    overwrite=True,
)

# Cohen's H will always use fixed groups
metric = FixedGroupNormCohensHParaphraseAccuracy()
global_target = {
    "fixed_group_norm_cohens_h_paraphrase_accuracy": -0.42,
    "score": -0.42,
    "score_name": "fixed_group_norm_cohens_h_paraphrase_accuracy",
    "score_ci_low": -1.0,
    "score_ci_high": 0.33,
    "fixed_group_norm_cohens_h_paraphrase_accuracy_ci_low": -1.0,
    "fixed_group_norm_cohens_h_paraphrase_accuracy_ci_high": 0.33,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_norm_cohens_h_paraphrase_accuracy",
    overwrite=True,
)


metric = FixedGroupNormCohensHParaphraseStringContainment()
global_target = {
    "fixed_group_norm_cohens_h_paraphrase_string_containment": -0.46,
    "score": -0.46,
    "score_name": "fixed_group_norm_cohens_h_paraphrase_string_containment",
    "score_ci_low": -0.5,
    "score_ci_high": -0.39,
    "fixed_group_norm_cohens_h_paraphrase_string_containment_ci_low": -0.5,
    "fixed_group_norm_cohens_h_paraphrase_string_containment_ci_high": -0.39,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_norm_cohens_h_paraphrase_string_containment",
    overwrite=True,
)


# Hedge's g will always use fixed groups
metric = FixedGroupNormHedgesGParaphraseAccuracy()
global_target = {
    "fixed_group_norm_hedges_g_paraphrase_accuracy": -0.35,
    "score": -0.35,
    "score_name": "fixed_group_norm_hedges_g_paraphrase_accuracy",
    "score_ci_low": -1.0,
    "score_ci_high": 0.02,
    "fixed_group_norm_hedges_g_paraphrase_accuracy_ci_low": -1.0,
    "fixed_group_norm_hedges_g_paraphrase_accuracy_ci_high": 0.02,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_norm_hedges_g_paraphrase_accuracy",
    overwrite=True,
)


metric = FixedGroupNormHedgesGParaphraseStringContainment()
global_target = {
    "fixed_group_norm_hedges_g_paraphrase_string_containment": -0.08,
    "score": -0.08,
    "score_name": "fixed_group_norm_hedges_g_paraphrase_string_containment",
    "score_ci_low": -0.1,
    "score_ci_high": -0.05,
    "fixed_group_norm_hedges_g_paraphrase_string_containment_ci_low": -0.1,
    "fixed_group_norm_hedges_g_paraphrase_string_containment_ci_high": -0.05,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_norm_hedges_g_paraphrase_string_containment",
    overwrite=True,
)

# absolute value of above metrics

metric = FixedGroupAbsvalNormCohensHParaphraseAccuracy()
global_target = {
    "fixed_group_absval_norm_cohens_h_paraphrase_accuracy": 0.65,
    "score": 0.65,
    "score_name": "fixed_group_absval_norm_cohens_h_paraphrase_accuracy",
    "score_ci_low": 0.33,
    "score_ci_high": 1.0,
    "fixed_group_absval_norm_cohens_h_paraphrase_accuracy_ci_low": 0.33,
    "fixed_group_absval_norm_cohens_h_paraphrase_accuracy_ci_high": 1.0,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_absval_norm_cohens_h_paraphrase_accuracy",
    overwrite=True,
)


metric = FixedGroupAbsvalNormCohensHParaphraseStringContainment()
global_target = {
    "fixed_group_absval_norm_cohens_h_paraphrase_string_containment": 0.46,
    "score": 0.46,
    "score_name": "fixed_group_absval_norm_cohens_h_paraphrase_string_containment",
    "score_ci_low": 0.39,
    "score_ci_high": 0.5,
    "fixed_group_absval_norm_cohens_h_paraphrase_string_containment_ci_low": 0.39,
    "fixed_group_absval_norm_cohens_h_paraphrase_string_containment_ci_high": 0.5,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_absval_norm_cohens_h_paraphrase_string_containment",
    overwrite=True,
)


metric = FixedGroupAbsvalNormHedgesGParaphraseAccuracy()
global_target = {
    "fixed_group_absval_norm_hedges_g_paraphrase_accuracy": 0.38,
    "score": 0.38,
    "score_name": "fixed_group_absval_norm_hedges_g_paraphrase_accuracy",
    "score_ci_low": 0.06,
    "score_ci_high": 1.0,
    "fixed_group_absval_norm_hedges_g_paraphrase_accuracy_ci_low": 0.06,
    "fixed_group_absval_norm_hedges_g_paraphrase_accuracy_ci_high": 1.0,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_accuracy,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_absval_norm_hedges_g_paraphrase_accuracy",
    overwrite=True,
)


metric = FixedGroupAbsvalNormHedgesGParaphraseStringContainment()
global_target = {
    "fixed_group_absval_norm_hedges_g_paraphrase_string_containment": 0.08,
    "score": 0.08,
    "score_name": "fixed_group_absval_norm_hedges_g_paraphrase_string_containment",
    "score_ci_low": 0.05,
    "score_ci_high": 0.1,
    "fixed_group_absval_norm_hedges_g_paraphrase_string_containment_ci_low": 0.05,
    "fixed_group_absval_norm_hedges_g_paraphrase_string_containment_ci_high": 0.1,
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
}


outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets_string_containment,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    metric,
    "metrics.robustness.fixed_group_absval_norm_hedges_g_paraphrase_string_containment",
    overwrite=True,
)

# TokenOverlap: example of a metric that has more than one score

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
    "num_of_instances": 15,
    "num_of_instances_in_group_group1": 5,
    "num_of_instances_in_group_group2": 5,
    "num_of_instances_in_group_group3": 4,
    "num_of_instances_in_group_group4": 1,
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
    task_data=task_data,
)

add_to_catalog(metric, "metrics.robustness.group_mean_token_overlap", overwrite=True)
