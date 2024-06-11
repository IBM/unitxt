from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Dictify, DuplicateBySubLists, Get, Wrap
from unitxt.dialog_operators import SerializeDialog
from unitxt.operators import CopyFields, ZipFieldValues
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="stanfordnlp/coqa"),
    preprocess_steps=[
        "splitters.small_no_test",
        Set(fields={"context_type": "story"}),
        ZipFieldValues(
            fields=["questions", "answers/input_text"],
            to_field="dialog",
        ),
        Dictify(field="dialog", with_keys=["user", "system"], process_every_value=True),
        DuplicateBySubLists(field="dialog"),
        Get(field="dialog", item=-1, to_field="last_turn"),
        CopyFields(
            field_to_field={"last_turn/user": "question", "last_turn/system": "answer"},
        ),
        Wrap(
            field="answer",
            inside="list",
            to_field="answers",
        ),
        SerializeDialog(
            field="dialog",
            to_field="context",
            context_field="story",
        ),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": ["1808.07042", "1704.04683", "1506.03340"],
        "flags": ["conversational-qa"],
        "language": "en",
        "language_creators": "found",
        "license": "other",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "1K<n<10K",
        "source_datasets": [
            "extended|race",
            "extended|cnn_dailymail",
            "extended|wikipedia",
            "extended|other",
        ],
        "task_categories": "question-answering",
        "task_ids": "extractive-qa",
    },
    __description__=(
        "CoQA is a large-scale dataset for building Conversational Question Answering systems. \n"
        "Our dataset contains 127k questions with answers, obtained from 8k conversations about text passages from seven diverse domains. The questions are conversational, and the answers are free-form text with their corresponding evidence highlighted in the passage. Supported Tasks and Leaderboards More Information Needed… See the full description on the dataset page: https://huggingface.co/datasets/stanfordnlp/coqa."
    ),
)

test_card(card)
add_to_catalog(card, "cards.coqa.qa", overwrite=True)

card = TaskCard(
    loader=LoadHF(path="stanfordnlp/coqa"),
    preprocess_steps=[
        "splitters.small_no_test",
        Set(fields={"context_type": "dialog", "completion_type": "response"}),
        ZipFieldValues(
            fields=["questions", "answers/input_text"],
            to_field="dialog",
        ),
        Dictify(field="dialog", with_keys=["user", "system"], process_every_value=True),
        DuplicateBySubLists(field="dialog"),
        SerializeDialog(
            field="dialog",
            to_field="context",
            context_field="story",
            last_response_to_field="completion",
        ),
    ],
    task="tasks.completion.abstractive",
    templates="templates.completion.abstractive.all",
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": ["1808.07042", "1704.04683", "1506.03340"],
        "flags": ["conversational-qa"],
        "language": "en",
        "language_creators": "found",
        "license": "other",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "1K<n<10K",
        "source_datasets": [
            "extended|race",
            "extended|cnn_dailymail",
            "extended|wikipedia",
            "extended|other",
        ],
        "task_categories": "question-answering",
        "task_ids": "extractive-qa",
    },
    __description__=(
        "CoQA is a large-scale dataset for building Conversational Question Answering systems. \n"
        "Our dataset contains 127k questions with answers, obtained from 8k conversations about text passages from seven diverse domains. The questions are conversational, and the answers are free-form text with their corresponding evidence highlighted in the passage. Supported Tasks and Leaderboards More Information Needed… See the full description on the dataset page: https://huggingface.co/datasets/stanfordnlp/coqa."
    ),
)

test_card(card)
add_to_catalog(card, "cards.coqa.completion", overwrite=True)
