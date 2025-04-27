import unitxt
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import DictToTuplesList, Wrap
from unitxt.loaders import LoadCSV
from unitxt.operators import (
    Copy,
)
from unitxt.stream_operators import JoinStreams
from unitxt.test_utils.card import test_card
from unitxt.tool_calling import ToTool

base_path = "https://raw.githubusercontent.com/ShishirPatil/gorilla/70b6a4a2144597b1f99d1f4d3185d35d7ee532a4/berkeley-function-call-leaderboard/data/"

with unitxt.settings.context(allow_unverified_code=True):
    card = TaskCard(
        loader=LoadCSV(
            files={"questions": base_path + "BFCL_v3_simple.json", "answers": base_path + "possible_answer/BFCL_v3_simple.json"},
            file_type="json",
            lines=True,
            data_classification_policy=["public"],
        ),
        preprocess_steps=[
            JoinStreams(left_stream="questions", right_stream ="answers", how="inner", on="id", new_stream_name="test" ),
            Copy(field="question/0/0/content", to_field="query"),
            ToTool(field="function/0", to_field="tool"),
            Wrap(field="tool", inside="list", to_field="tools"),
            DictToTuplesList(field="ground_truth/0", to_field="call_tuples"),
            Copy(field="call_tuples/0/0", to_field="call/name"),
            Copy(field="call_tuples/0/1", to_field="call/arguments"),
        ],
        task="tasks.tool_calling.supervised",
        templates=["templates.tool_calling.base"],
        __description__=(
            """gg"""
        ),
        __tags__={
            "annotations_creators": "expert-generated",
            "language": ["en"],
            "license": "cc-by-4.0",
            "size_categories": ["10K<n<100K"],
            "task_categories": [
                "question-answering",
                "multiple-choice",
                "reading-comprehension",
            ],
            "multilinguality": "monolingual",
            "task_ids": ["extractive-qa", "reading-comprehension"],
        },
    )

    # Test and add the card to the catalog
    test_card(card, strict=False)
    add_to_catalog(card, "cards.bfcl.simple_v3", overwrite=True)
