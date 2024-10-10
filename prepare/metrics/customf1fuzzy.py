from unitxt import add_to_catalog
from unitxt.metrics import FuzzyNer
from unitxt.test_utils.metrics import test_metric

metric = FuzzyNer()
# Test1 single line single class
# 0.1 simple case, multi examples
predictions = [
    [("Amir", "Person"), ("Yaron", "Person")],
    [("Ran", "Person"), ("Yonatan", "Person")],
]
references = [[[("Yaron", "Person"), ("Ran", "Person")]], [[("Yonatan", "Person")]]]

instance_targets = [
    {
        "f1_Person": 0.5,
        "f1_macro": 0.5,
        "in_classes_support": 1.0,
        "f1_micro": 0.5,
        "recall_micro": 0.5,
        "recall_macro": 0.5,
        "precision_micro": 0.5,
        "precision_macro": 0.5,
        "score": 0.5,
        "score_name": "f1_micro",
    },
    {
        "f1_Person": 0.67,
        "f1_macro": 0.67,
        "in_classes_support": 1.0,
        "f1_micro": 0.67,
        "recall_micro": 1.0,
        "recall_macro": 1.0,
        "precision_micro": 0.5,
        "precision_macro": 0.5,
        "score": 0.67,
        "score_name": "f1_micro",
    },
]
global_target = {
    "f1_Person": 0.57,
    "f1_macro": 0.57,
    "in_classes_support": 1.0,
    "f1_micro": 0.57,
    "recall_micro": 0.67,
    "recall_macro": 0.67,
    "precision_micro": 0.5,
    "precision_macro": 0.5,
    "score": 0.57,
    "score_name": "f1_micro",
    "f1_micro_ci_low": 0.57,
    "f1_micro_ci_high": 0.57,
    "score_ci_low": 0.57,
    "score_ci_high": 0.57,
    "num_of_evaluated_instances": 2,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

# 1.1 Extra characters in prediction due to ocr issues
predictions = [[("Amir,", "Person"), ("Yaroh", "Person")]]
references = [[[("Yaron", "Person"), ("Ran", "Person"), ("Yonatan", "Person")]]]
# Precision = 1/2, Recall = 1/3, F1 = 2 * 1/2 * 1/3 / (1/2 + 1/3) = 0.4
instance_targets = [
    {
        "recall_micro": 0.33,
        "recall_macro": 0.33,
        "precision_micro": 0.5,
        "precision_macro": 0.5,
        "f1_Person": 0.4,
        "f1_macro": 0.4,
        "in_classes_support": 1.0,
        "f1_micro": 0.4,
        "score": 0.4,
        "score_name": "f1_micro",
    }
]
global_target = {
    "recall_micro": 0.33,
    "recall_macro": 0.33,
    "precision_micro": 0.5,
    "precision_macro": 0.5,
    "f1_Person": 0.4,
    "f1_macro": 0.4,
    "in_classes_support": 1.0,
    "f1_micro": 0.4,
    "score": 0.4,
    "score_name": "f1_micro",
    "num_of_evaluated_instances": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
# 1.2 more then one instance of an element
predictions = [[("Amir", "Person"), ("Yaroh#", "Person"), ("Yaron", "Person")]]
references = [[[("Yaron", "Person"), ("Ran", "Person"), ("Yonatan", "Person")]]]
# Precision = 1/3, Recall = 1/3, F1 = 2 * 1/3 * 1/3 / (1/3 + 1/3) = 0.333333
instance_targets = [
    {
        "recall_micro": 0.33,
        "recall_macro": 0.33,
        "precision_micro": 0.33,
        "precision_macro": 0.33,
        "f1_Person": 0.33,
        "f1_macro": 0.33,
        "in_classes_support": 1.0,
        "f1_micro": 0.33,
        "score": 0.33,
        "score_name": "f1_micro",
    }
]
global_target = {
    "recall_micro": 0.33,
    "recall_macro": 0.33,
    "precision_micro": 0.33,
    "precision_macro": 0.33,
    "f1_Person": 0.33,
    "f1_macro": 0.33,
    "in_classes_support": 1.0,
    "f1_micro": 0.33,
    "score": 0.33,
    "score_name": "f1_micro",
    "num_of_evaluated_instances": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
# 1.3 class with prediction match less than threshold
predictions = [[("Yar", "Person"), ("atan", "Person")]]
references = [[[("Yaron", "Person"), ("Ran", "Person"), ("Yonatan", "Person")]]]
# Precision = 0/0=(by def for prediction)=0, Recall = 0/3, F1 = 2 * 1 * 0 / (1 + 0) = 0
instance_targets = [
    {
        "recall_micro": 0.0,
        "recall_macro": 0.0,
        "precision_micro": 0.0,
        "precision_macro": 0.0,
        "f1_Person": 0.0,
        "f1_macro": 0.0,
        "in_classes_support": 1.0,
        "f1_micro": 0.0,
        "score": 0.0,
        "score_name": "f1_micro",
    }
]
global_target = {
    "recall_micro": 0.0,
    "recall_macro": 0.0,
    "precision_micro": 0.0,
    "precision_macro": 0.0,
    "f1_Person": 0.0,
    "f1_macro": 0.0,
    "in_classes_support": 1.0,
    "f1_micro": 0.0,
    "score": 0.0,
    "score_name": "f1_micro",
    "num_of_evaluated_instances": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
# 1.4 class with multiple predictions where one matches the threshold and other doesn't
predictions = [[("Yar0h#", "Person"), ("Yonatan", "Person")]]
references = [[[("Yaron", "Person"), ("Ran", "Person"), ("Yonatan", "Person")]]]
# Person: Precision = 3/3, Recall = 3/4, F1 = 2 * 1 * 0.75 / (1 + 0.75) = 0.8571
# Location: Precision = 1/2, Recall = 1/2, F1 = 0.5
# Org (OOD): Precision = 0/3, Recall = 0/0 = 1(!), F1 = 0
instance_targets = [
    {
        "recall_micro": 0.33,
        "recall_macro": 0.33,
        "precision_micro": 0.5,
        "precision_macro": 0.5,  # Only on indomain classes
        "f1_Person": 0.4,
        "f1_macro": 0.4,
        "in_classes_support": 1.0,
        "f1_micro": 0.4,
        "score": 0.4,
        "score_name": "f1_micro",
    },
]
global_target = {
    "recall_micro": 0.33,
    "recall_macro": 0.33,
    "precision_micro": 0.5,
    "precision_macro": 0.5,  # Only on indomain classes
    "f1_Person": 0.4,
    "f1_macro": 0.4,
    "in_classes_support": 1.0,
    "f1_micro": 0.4,
    "score": 0.4,
    "score_name": "f1_micro",
    "num_of_evaluated_instances": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)


# 1.5 class in reference and not in prediction

# A (OOD): Precision = 0/1 = 0 , Recall = 0/0=1, F1 = 0
# B: Precision = 1, Recall = 0.5, F1 = 0.67
# C: Precision = 1/1, Recall = 1/1, F1 = 1
# D: Precision = 0/0 = 1 , Recall = 0, F1 = 0
predictions = [
    [
        ("a", "A"),
        ("b", "B"),
        ("c", "C"),
    ]
]
references = [
    [
        [
            ("a", "B"),
            ("b", "B"),
            ("c", "C"),
            ("d", "D"),
        ]
    ]
]

instance_targets = [
    {
        "recall_micro": 0.5,
        "recall_macro": 0.5,
        "precision_micro": 0.67,
        "precision_macro": 0.67,
        "f1_D": 0.0,
        "f1_C": 1.0,
        "f1_B": 0.67,
        "f1_macro": 0.56,
        "in_classes_support": 0.67,
        "f1_micro": 0.57,
        "score": 0.57,
        "score_name": "f1_micro",
    },
]
global_target = {
    "recall_micro": 0.5,
    "recall_macro": 0.5,
    "precision_micro": 0.67,
    "precision_macro": 0.67,
    "f1_D": 0.0,
    "f1_C": 1.0,
    "f1_B": 0.67,
    "f1_macro": 0.56,
    "in_classes_support": 0.67,
    "f1_micro": 0.57,
    "score": 0.57,
    "score_name": "f1_micro",
    "num_of_evaluated_instances": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

# 1.6 all predictions are out of domain

predictions = [
    [
        ("a", "A"),
    ]
]
references = [
    [
        [
            ("b", "B"),
        ]
    ]
]

instance_targets = [
    {
        "recall_micro": 0.0,
        "recall_macro": 0.0,
        "precision_micro": 0.0,
        "precision_macro": 0.0,
        "f1_B": 0.0,
        "f1_macro": 0.0,
        "in_classes_support": 0.0,
        "f1_micro": 0.0,
        "score": 0.0,
        "score_name": "f1_micro",
    },
]
global_target = {
    "recall_micro": 0.0,
    "recall_macro": 0.0,
    "precision_micro": 0.0,
    "precision_macro": 0.0,
    "f1_B": 0.0,
    "f1_macro": 0.0,
    "in_classes_support": 0.0,
    "f1_micro": 0.0,
    "score": 0.0,
    "score_name": "f1_micro",
    "num_of_evaluated_instances": 1,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)


add_to_catalog(metric, "metrics.fuzzyner", overwrite=True)
