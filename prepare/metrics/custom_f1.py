from src.unitxt import add_to_catalog
from src.unitxt.metrics import NER
from src.unitxt.test_utils.metrics import test_metric

metric = NER()
# Test1 single line single class
# 0.1 simple case, multi examples
predictions = [[("Amir", "Person"), ("Yaron", "Person")], [("Ran", "Person"), ("Yonatan", "Person")]]
references = [[("Yaron", "Person"), ("Ran", "Person")], [("Yonatan", "Person")]]
# precision = 1/2, recall 1/2
# Precision = 1/2, Recall = 1/1,
# total precision = 2/4, recall = 2/3
# F1 = 2 * 1/2 * 2/3 / (1/2 + 2/3) = 0.57
instance_targets = [
    {"f1_Person": 0.5, "f1_macro": 0.5, "in_classes_support": 1.0, "f1_micro": 0.5, "score": 0.5},
    {"f1_Person": 0.67, "f1_macro": 0.67, "in_classes_support": 1.0, "f1_micro": 0.67, "score": 0.67},
]
global_target = {"f1_Person": 0.57, "f1_macro": 0.57, "in_classes_support": 1.0, "f1_micro": 0.57, "score": 0.57}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

# 1.1 simple case
predictions = [[("Amir", "Person"), ("Yaron", "Person")]]
references = [[("Yaron", "Person"), ("Ran", "Person"), ("Yonatan", "Person")]]
# Precision = 1/2, Recall = 1/3, F1 = 2 * 1/2 * 1/3 / (1/2 + 1/3) = 0.4
instance_targets = [{"f1_Person": 0.4, "f1_macro": 0.4, "in_classes_support": 1.0, "f1_micro": 0.4, "score": 0.4}]
global_target = {"f1_Person": 0.4, "f1_macro": 0.4, "in_classes_support": 1.0, "f1_micro": 0.4, "score": 0.4}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
# 1.2 more then one instance of an element
predictions = [[("Amir", "Person"), ("Yaron", "Person"), ("Yaron", "Person")]]
references = [[("Yaron", "Person"), ("Ran", "Person"), ("Yonatan", "Person")]]
# Precision = 1/3, Recall = 1/3, F1 = 2 * 1/3 * 1/3 / (1/3 + 1/3) = 0.333333
instance_targets = [{"f1_Person": 0.33, "f1_macro": 0.33, "in_classes_support": 1.0, "f1_micro": 0.33, "score": 0.33}]
global_target = {"f1_Person": 0.33, "f1_macro": 0.33, "in_classes_support": 1.0, "f1_micro": 0.33, "score": 0.33}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
# 1.3 class with no predictions
predictions = [[]]
references = [[("Yaron", "Person"), ("Ran", "Person"), ("Yonatan", "Person")]]
# Precision = 0/0=(by def for prediction)=1, Recall = 0/3, F1 = 2 * 1 * 0 / (1 + 0) = 0
instance_targets = [{"f1_Person": 0.0, "f1_macro": 0.0, "in_classes_support": 1.0, "f1_micro": 0.0, "score": 0.0}]
global_target = {"f1_Person": 0.0, "f1_macro": 0.0, "in_classes_support": 1.0, "f1_micro": 0.0, "score": 0.0}

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
        ("Dalia", "Person"),
        ("Amir", "Person"),
        ("Yaron", "Person"),
        ("Ramat-Gan", "Location"),
        ("Ramat-Gan", "Location"),
        ("IBM", "Org"),
        ("CIA", "Org"),
        ("FBI", "Org"),
    ]
]
references = [
    [
        ("Amir", "Person"),
        ("Yaron", "Person"),
        ("Dalia", "Person"),
        ("Naftali", "Person"),
        ("Ramat-Gan", "Location"),
        ("Givataaim", "Location"),
    ]
]
# Person: Precision = 3/3, Recall = 3/4, F1 = 2 * 1 * 0.75 / (1 + 0.75) = 0.8571
# Location: Precision = 1/2, Recall = 1/2, F1 = 0.5
# Person: Precision = 0/3, Recall = 0/0 = 1(!), F1 = 0
instance_targets = [
    {
        "f1_Location": 0.5,
        "f1_Person": 0.86,
        "f1_macro": 0.68,
        "in_classes_support": 0.62,
        "f1_micro": 0.57,
        "score": 0.57,
    },
]
global_target = {
    "f1_Location": 0.5,
    "f1_Person": 0.86,
    "f1_macro": 0.68,
    "in_classes_support": 0.62,
    "f1_micro": 0.57,
    "score": 0.57,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)


# 1.5 class in prediction and not in reference
predictions = [
    [
        ("a", "A"),
        ("b", "B"),
        ("c", "C"),
    ]
]
references = [
    [
        ("a", "B"),
        ("b", "B"),
        ("c", "C"),
        ("d", "D"),
    ]
]

instance_targets = [
    {
        "f1_D": 0.0,
        "f1_C": 1.0,
        "f1_B": 0.67,
        "f1_macro": 0.56,
        "in_classes_support": 0.67,
        "f1_micro": 0.57,
        "score": 0.57,
    },
]
global_target = {
    "f1_D": 0.0,
    "f1_C": 1.0,
    "f1_B": 0.67,
    "f1_macro": 0.56,
    "in_classes_support": 0.67,
    "f1_micro": 0.57,
    "score": 0.57,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)


add_to_catalog(metric, "metrics.ner", overwrite=True)
