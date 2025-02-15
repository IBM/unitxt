from typing import Any, Dict

from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import DuplicateBySubLists, Pop, Wrap
from unitxt.dialog_operators import ToDialog
from unitxt.operator import InstanceOperator
from unitxt.operators import AddID, Copy, ZipFieldValues
from unitxt.test_utils.card import test_card


class Pass(InstanceOperator):
    def process(
        self, instance: Dict[str, Any], stream_name: str | None = None
    ) -> Dict[str, Any]:
        return instance


card = TaskCard(
    loader=LoadHF(path="stanfordnlp/coqa"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddID(),
        Copy(field="id", to_field="conversation/id"),
        ZipFieldValues(
            fields=["questions", "answers/input_text"],
            to_field="dialog",
        ),
        DuplicateBySubLists(field="dialog"),
        ToDialog(field="dialog"),
        Pop(field="dialog", item=-1, to_field="last_turn"),
        Copy(
            field_to_field={"last_turn/content": "answer", "story": "context"},
        ),
        Wrap(
            field="answer",
            inside="list",
            to_field="answers",
        ),
        Copy(field="dialog", to_field="conversation/dialog"),
    ],
    task="tasks.qa.extractive.multi_turn",
    templates=["templates.qa.multi_turn.with_context.simple"],
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
add_to_catalog(card, "cards.coqa.multi_turn", overwrite=True)
