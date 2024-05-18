from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import IndexOf, RenameFields
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="openbookqa"),
    preprocess_steps=[
        RenameFields(
            field_to_field={"choices/text": "choices_text", "choices/label": "labels"},
        ),
        RenameFields(
            field_to_field={"choices_text": "choices", "question_stem": "question"},
        ),
        IndexOf(search_in="labels", index_of="answerKey", to_field="answer"),
    ],
    task="tasks.qa.multiple_choice.open",
    templates="templates.qa.multiple_choice.open.all",
    __tags__={
        "dataset_info_tags": [
            "task_categories:question-answering",
            "task_ids:open-domain-qa",
            "annotations_creators:crowdsourced",
            "annotations_creators:expert-generated",
            "language_creators:expert-generated",
            "multilinguality:monolingual",
            "size_categories:1K<n<10K",
            "source_datasets:original",
            "language:en",
            "license:unknown",
            "croissant",
            "region:us",
        ]
    },
)
test_card(card, strict=False)
add_to_catalog(card, "cards.openbook_qa", overwrite=True)
