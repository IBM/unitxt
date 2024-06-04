from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    AddFields,
    IndexOf,
    ListFieldValues,
    RenameFields,
    ShuffleFieldValues,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="sciq"),
    preprocess_steps=[
        ListFieldValues(
            fields=["distractor1", "distractor2", "distractor3", "correct_answer"],
            to_field="choices",
        ),
        ShuffleFieldValues(field="choices"),
        IndexOf(search_in="choices", index_of="correct_answer", to_field="answer"),
        RenameFields(
            field_to_field={"support": "context"},
        ),
        AddFields({"context_type": "paragraph"}),
    ],
    task="tasks.qa.multiple_choice.with_context",
    templates="templates.qa.multiple_choice.with_context.all",
    __tags__={
        "annotations_creators": "no-annotation",
        "flags": ["croissant"],
        "language": "en",
        "language_creators": "crowdsourced",
        "license": "cc-by-nc-3.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "question-answering",
        "task_ids": "closed-domain-qa",
    },
    __description__=(
        'Dataset Card for "sciq" Dataset Summary The SciQ dataset contains 13,679 crowdsourced science exam questions about Physics, Chemistry and Biology, among others. The questions are in multiple-choice format with 4 answer options each. For the majority of the questions, an additional paragraph with supporting evidence for the correct answer is provided. Supported Tasks and Leaderboards More Information Needed Languages More Information… See the full description on the dataset page: https://huggingface.co/datasets/allenai/sciq.'
    ),
)
test_card(card, strict=False)
add_to_catalog(card, "cards.sciq", overwrite=True)
