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
        "annotations_creators": ["crowdsourced", "expert-generated"],
        "language": "en",
        "language_creators": "expert-generated",
        "license": "unknown",
        "multilinguality": "monolingual",
        "region": "us",
        "singletons": ["croissant"],
        "size_categories": "1K<n<10K",
        "source_datasets": "original",
        "task_categories": "question-answering",
        "task_ids": "open-domain-qa",
    },
    __description__=(
        "Dataset Card for OpenBookQA\n"
        "Dataset Summary\n"
        "OpenBookQA aims to promote research in advanced question-answering, probing a deeper understanding of both the topic\n"
        "(with salient facts summarized as an open book, also provided with the dataset) and the language it is expressed in. In\n"
        "particular, it contains questions that require multi-step reasoning, use of additional common and commonsense knowledge,\n"
        "and rich text comprehension.\n"
        "OpenBookQA is a new kind of… See the full description on the dataset page: https://huggingface.co/datasets/allenai/openbookqa."
    ),
)
test_card(card, strict=False)
add_to_catalog(card, "cards.openbook_qa", overwrite=True)
