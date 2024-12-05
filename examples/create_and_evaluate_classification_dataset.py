from unitxt import get_logger
from unitxt.api import create_and_evaluate_dataset
from unitxt.text_utils import print_dict

logger = get_logger()


classes = ["positive", "negative"]


# Set up question answer pairs in a dictionary
dataset = [
    {"text": "I am happy.", "label": "positive", "classes": classes},
    {"text": "It was a great movie.", "label": "positive", "classes": classes},
    {"text": "I never felt so bad", "label": "negative", "classes": classes},
]

predictions = ["Positive.", "negative.", "negative"]

evaluated_dataset = create_and_evaluate_dataset(
    task="tasks.classification.multi_class",
    predictions=predictions,
    postprocessors=["processors.take_first_word", "processors.lower_case"],
    data=dataset,
)

# Print results
for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "prediction",
            "references",
            "score",
        ],
    )
