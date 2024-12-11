from typing import List, Union

from unitxt.catalog import add_to_catalog
from unitxt.task import Task
from unitxt.types import Audio, Dialog, Image, Table, Text

add_to_catalog(
    Task(
        __description__="""This is binary text classification task.
        The 'class' is the name of the class we classify for and must be the same in all instances.
        The 'text_type' is an optional field that defines the type of text we classify (e.g. "document", "review", etc.).
        This can be used by the template to customize the prompt.

        The expected output is a list which is either an empty list [] or a list with a single element with the class name.

        The default reported metrics are the classical f1_micro, f1_macro and accuracy.
        """,
        input_fields={"text": str, "text_type": str, "class": str},
        reference_fields={"class": str, "label": List[str]},
        prediction_type=List[str],
        metrics=[
            "metrics.f1_micro_multi_label",
            "metrics.f1_macro_multi_label",
            "metrics.accuracy",
        ],
        augmentable_inputs=["text"],
        defaults={"text_type": "text"},
    ),
    "tasks.classification.binary",
    overwrite=True,
)

add_to_catalog(
    Task(
        __description__="""This is binary text classification task where the labels are provided as 0 and 1.

The 'class' is the name of the class we classifify and must be the same in all instances.
The 'text_type' is an optional field that defines the type of text we classify (e.g. "document", "review", etc.).
This can be used by the template to customize the prompt.

The default reported metrics are the classifical f1_micro (accuracy).
        """,
        input_fields={"text": str, "text_type": str, "class": str},
        reference_fields={"class": str, "label": int},
        prediction_type=float,
        metrics=[
            "metrics.accuracy",
            "metrics.f1_binary",
        ],
        augmentable_inputs=["text"],
        defaults={"text_type": "text"},
    ),
    "tasks.classification.binary.zero_or_one",
    overwrite=True,
)

add_to_catalog(
    Task(
        __description__="""This is multi label text classification task.
The set of 'classes' we want to classify to is provided as a list of strings.

The 'text_type' is an optional field that defines the type of text we classify (e.g. "document", "review", etc.).
This can be used by the template to customize the prompt.

The 'type_of_class' is a field that the defines the type of classes  (e.g. "emotions", "risks")

The 'classes' , 'type_of_classes' and 'text_type' should be the same on all instances.

The expected output is a list of classes that correspond to the given text (could be an empty list.
The default reported metrics are the classical f1_micro, f1_macro and accuracy.
""",
        input_fields={
            "text": str,
            "text_type": str,
            "classes": List[str],
            "type_of_classes": str,
        },
        reference_fields={"labels": List[str]},
        prediction_type=List[str],
        metrics=[
            "metrics.f1_micro_multi_label",
            "metrics.accuracy",
            "metrics.f1_macro_multi_label",
        ],
        augmentable_inputs=["text"],
        defaults={"text_type": "text", "type_of_classes": "classes"},
        default_template="templates.classification.multi_label.title",
    ),
    "tasks.classification.multi_label",
    overwrite=True,
)

add_to_catalog(
    Task(
        __description__="""This is multi class text classification task.

The set of 'classes' we want to classify to is provided as a list of strings.

The 'text_type' is an optional field that defines the type of text we classify (e.g. "document", "review", etc.).
The 'type_of_class' is an oiptional field that the defines the type of classification we perform (e.g. "sentiment", "harm", "risk" etc..)
The 'text_type' and 'type_of_class' fields can be used by the template to customize the prompt.

The default reported metrics are the classical f1_micro (equivalent to accuracy for multi class classification), and f1_macro.

""",
        input_fields={
            "text": str,
            "text_type": str,
            "classes": List[str],
            "type_of_class": str,
        },
        reference_fields={"label": str},
        prediction_type=str,
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        augmentable_inputs=["text"],
        defaults={"text_type": "text", "type_of_class": "class"},
        default_template="templates.classification.multi_class.title",
    ),
    "tasks.classification.multi_class",
    overwrite=True,
)

add_to_catalog(
    Task(
        __description__="""This is a special case of multi class text classification task, in which we classify the relation between two texts.
For example, whether one text entails another.
The inputs are provided in "text_a" and "text_a"
The set of 'classes' is a list of option of the relationship (e.g. "entailment", "contradiction", "neutral")
The 'text_a_type' and 'text_type" are optional fields that defines the type of text we classify (e.g. "document", "review", etc.).
The 'type_of_relation' is a required field that the defines the type of relation we identify (e.g. "entailment")
The 'text_a_type','text_b_type' and 'type_of_relation' fields can be used by the template to customize the prompt.

The default reported metrics are the classical f1_micro (equivalent to accuracy for multi class classification), and f1_macro.

""",
        input_fields={
            "text_a": Union[Text, Image, Audio, Table, Dialog],
            "text_a_type": str,
            "text_b": str,
            "text_b_type": str,
            "classes": List[str],
            "type_of_relation": str,
        },
        reference_fields={"label": str},
        prediction_type=str,
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        augmentable_inputs=["text_a", "text_b"],
        defaults={"text_a_type": "first text", "text_b_type": "second text"},
        default_template="templates.classification.multi_class.title",
    ),
    "tasks.classification.multi_class.relation",
    overwrite=True,
)


add_to_catalog(
    Task(
        __description__="""This is a special case of multi class text classification task, in which we classify a given text to a set of topics.
The only difference from 'tasks.classification.multi_class', is that the addition of 'classes_descriptions' field,
which is used by the template to add a description for each class.
""",
        input_fields={
            "text": str,
            "text_type": str,
            "classes": List[str],
            "type_of_class": str,
            "classes_descriptions": str,
        },
        reference_fields={"label": str},
        prediction_type=str,
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        augmentable_inputs=["text"],
        defaults={"text_type": "text", "type_of_class": "class"},
    ),
    "tasks.classification.multi_class.with_classes_descriptions",
    overwrite=True,
)

add_to_catalog(
    Task(
        __description__="""This is a special case of multi class text classification task, in which we classify a given text to a set of topics.
The only difference from tasks.classification.multi_class, is that the the 'type_of_class' is set to 'topic'.
""",
        input_fields={
            "text": str,
            "text_type": str,
            "classes": List[str],
            "type_of_class": str,
        },
        reference_fields={"label": str},
        prediction_type=str,
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        augmentable_inputs=["text"],
        defaults={"text_type": "text", "type_of_class": "topic"},
    ),
    "tasks.classification.multi_class.topic_classification",
    overwrite=True,
)
