from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={"text": "str", "text_type": "str", "sentiment_class": "str"},
        reference_fields={
            "spans_starts": "List[int]",
            "spans_ends": "List[int]",
            "text": "List[str]",
            "labels": "List[str]",
        },
        prediction_type="List[Tuple[str,str]]",
        metrics=["metrics.ner"],
        augmentable_inputs=["text"],
        defaults={"text_type": "text"},
    ),
    "tasks.targeted_sentiment_extraction.single_sentiment_class",
    overwrite=True,
)

add_to_catalog(
    Task(
        input_fields={"text": "str", "text_type": "str"},
        reference_fields={
            "spans_starts": "List[int]",
            "spans_ends": "List[int]",
            "text": "List[str]",
            "labels": "List[str]",
        },
        prediction_type="List[Tuple[str,str]]",
        metrics=["metrics.ner"],
        augmentable_inputs=["text"],
        defaults={"text_type": "text"},
    ),
    "tasks.targeted_sentiment_extraction.all_sentiment_classes",
    overwrite=True,
)
