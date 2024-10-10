from unitxt.blocks import (
    LoadHF,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import Copy, Set
from unitxt.templates import MultiReferenceTemplate
from unitxt.test_utils.card import test_card

card = TaskCard(
    # Adjust the num_proc value according to the number of CPU cores available for faster loading
    loader=LoadHF(
        path="wikitablequestions", data_classification_policy=["public"], num_proc=10
    ),
    preprocess_steps=[
        Set({"context_type": "table"}),
        Copy(field="table", to_field="context"),
    ],
    task="tasks.qa.with_context.extractive[metrics=[metrics.f1_strings, metrics.unsorted_list_exact_match]]",
    templates=[
        MultiReferenceTemplate(
            input_format="Based on this {context_type}: {context}\nAnswer the question: {question}",
            references_field="answers",
            postprocessors=[
                # "processors.to_list_by_comma_space",
                "processors.str_to_float_format",
            ],
        ),
    ],
    __description__=(
        "This WikiTableQuestions dataset is a large-scale dataset for the task of question answering on semi-structured tables… See the full description on the dataset page: https://huggingface.co/datasets/wikitablequestions"
    ),
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": "1508.00305",
        "flags": ["table-question-answering"],
        "language": "en",
        "language_creators": "found",
        "license": "cc-by-4.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "question-answering",
    },
)

test_card(card, strict=False)
add_to_catalog(card, "cards.wikitq", overwrite=True)
