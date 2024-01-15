from unitxt import add_to_catalog
from unitxt.blocks import FormTask

add_to_catalog(
    FormTask(
        inputs=["text", "text_type", "sentiment_class"],
        outputs=["spans_starts", "spans_ends", "text", "labels"],
        metrics=["metrics.ner"],
        augmentable_inputs=["text"],
    ),
    "tasks.targeted_sentiment_extraction.single_sentiment_class",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs=["text", "text_type"],
        outputs=["spans_starts", "spans_ends", "text", "labels"],
        metrics=["metrics.ner"],
        augmentable_inputs=["text"],
    ),
    "tasks.targeted_sentiment_extraction.all_sentiment_classes",
    overwrite=True,
)
