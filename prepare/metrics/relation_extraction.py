import sys

from unitxt import add_to_catalog

sys.path.append("/Users/pklocek/fm_eval_workspace/unitxt/src/unitxt/metrics.py")
from unitxt.metrics import RelationExtraction
from unitxt.test_utils.metrics import test_metric

metric = RelationExtraction()

predictions = [[("Amir", "employedBy", "IBM"), ("Yaron", "employedBy", "IBM")]]
references = [
    [
        [
            ("Yaron", "employedBy", "IBM"),
            ("Ran", "employedBy", "IBM"),
            ("Yonatan", "employedBy", "IBM"),
        ]
    ]
]
# Precision = 1/2, Recall = 1/3, F1 = 2 * 1/2 * 1/3 / (1/2 + 1/3) = 0.4
instance_targets = [
    {
        "recall_micro": 0.33,
        "recall_macro": 0.33,
        "precision_micro": 0.5,
        "precision_macro": 0.5,
        "f1_employedBy": 0.4,
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
    "f1_employedBy": 0.4,
    "f1_macro": 0.4,
    "in_classes_support": 1.0,
    "f1_micro": 0.4,
    "score": 0.4,
    "score_name": "f1_micro",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)


# 1.2 more then one instance of an element
predictions = [
    [
        ("Amir", "employedBy", "IBM"),
        ("Yaron", "employedBy", "IBM"),
        ("Yaron", "employedBy", "IBM"),
    ]
]
references = [
    [
        [
            ("Yaron", "employedBy", "IBM"),
            ("Ran", "employedBy", "IBM"),
            ("Yonatan", "employedBy", "IBM"),
        ]
    ]
]
# Precision = 1/2, Recall = 1/3, F1 = 2 * 1/2 * 1/3 / (1/2 + 1/3) = 0.4
instance_targets = [
    {
        "recall_micro": 0.33,
        "recall_macro": 0.33,
        "precision_micro": 0.33,
        "precision_macro": 0.33,
        "f1_employedBy": 0.33,
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
    "f1_employedBy": 0.33,
    "f1_macro": 0.33,
    "in_classes_support": 1.0,
    "f1_micro": 0.33,
    "score": 0.33,
    "score_name": "f1_micro",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
# 1.3 class with no predictions
predictions = [[]]
references = [
    [
        [
            ("Yaron", "employedBy", "IBM"),
            ("Ran", "employedBy", "IBM"),
            ("Yonatan", "employedBy", "IBM"),
        ]
    ]
]
# Precision = 0/0=(by def for prediction)=0, Recall = 0/3, F1 = 2 * 1 * 0 / (1 + 0) = 0
instance_targets = [
    {
        "recall_micro": 0.0,
        "recall_macro": 0.0,
        "precision_micro": 0.0,
        "precision_macro": 0.0,
        "f1_employedBy": 0.0,
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
    "f1_employedBy": 0.0,
    "f1_macro": 0.0,
    "in_classes_support": 1.0,
    "f1_micro": 0.0,
    "score": 0.0,
    "score_name": "f1_micro",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
# 1.4 multi classes multi examples
predictions = [
    [
        ("Amir", "employedBy", "IBM"),
        ("Kyle", "employedBy", "Google"),
        ("Jenna", "employedBy", "Exxon"),
        ("Rami", "managerOf", "Tom"),
        ("Rami", "managerOf", "Tom"),
        ("James", "basedIn", "Chicago"),
        ("Abhishek", "basedIn", "Bangalore"),
        ("Zuzia", "basedIn", "Cracow"),
    ]
]
references = [
    [
        [
            ("Jenna", "employedBy", "Exxon"),
            ("Amir", "employedBy", "IBM"),
            ("Kyle", "employedBy", "Google"),
            ("George", "employedBy", "Bloomberg"),
            ("Rami", "managerOf", "Tom"),
            ("Annie", "managerOf", "Trevor"),
        ]
    ]
]
# Person: Precision = 3/3, Recall = 3/4, F1 = 2 * 1 * 0.75 / (1 + 0.75) = 0.8571
# Location: Precision = 1/2, Recall = 1/2, F1 = 0.5
# Org (OOD): Precision = 0/3, Recall = 0/0 = 1(!), F1 = 0
instance_targets = [
    {
        "recall_micro": 0.67,
        "recall_macro": 0.62,
        "precision_micro": 0.5,
        "precision_macro": 0.75,  # Only on indomain classes
        "f1_managerOf": 0.5,
        "f1_employedBy": 0.86,
        "f1_macro": 0.68,
        "in_classes_support": 0.62,
        "f1_micro": 0.57,
        "score": 0.57,
        "score_name": "f1_micro",
    },
]
global_target = {
    "recall_micro": 0.67,
    "recall_macro": 0.62,
    "precision_micro": 0.5,
    "precision_macro": 0.75,
    "f1_managerOf": 0.5,
    "f1_employedBy": 0.86,
    "f1_macro": 0.68,
    "in_classes_support": 0.62,
    "f1_micro": 0.57,
    "score": 0.57,
    "score_name": "f1_micro",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.relation_extraction", overwrite=True)
