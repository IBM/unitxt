import unitxt
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadCSV
from unitxt.operators import (
    Copy,
    ExecuteExpression,
    RecursiveReplace,
)
from unitxt.stream_operators import JoinStreams
from unitxt.test_utils.card import test_card

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
            Copy(field="function", to_field="tools"),
            RecursiveReplace(key="type", map_values={"dict": "object", "float": "number", "tuple": "array"}, remove_values=["any"]),
            # Process ground truth data in this dataset, which is a provided as a list of options per field,
            # and convert it into a list of explicit tool calls
            #
            #[{"geometry.circumference": {"radius": [3], "units": ["cm", "m"]}}]}
            # becomes:
            # [{"name": "geometry.circumference", "arguments" : {"radius": 3, "units": "cm"}},
            #  {"name": "geometry.circumference", "arguments" : {"radius": 3, "units": "m"}}]

            ExecuteExpression(expression='[{"name": k, "arguments": dict(zip(v.keys(), vals))} for d in ground_truth for k, v in d.items() for vals in itertools.product(*v.values())]',
                              to_field="reference_calls", imports_list=["itertools"])
        ],
        task="tasks.tool_calling.supervised",
        templates=["templates.tool_calling.base"],
        __description__=(
            """The Berkeley function calling leaderboard is a live leaderboard to evaluate the ability of different LLMs to call functions (also referred to as tools). We built this dataset from our learnings to be representative of most users' function calling use-cases, for example, in agents, as a part of enterprise workflows, etc. To this end, our evaluation dataset spans diverse categories, and across multiple languages."""
        ),
        __title__="Berkeley Function Calling Leaderboard - Simple V3",
        __tags__={
            "annotations_creators": "expert-generated",
            "language": ["en"],
            "license": "apache-2.0",
            "size_categories": ["10K<n<100K"],
            "task_categories": [
                "question-answering",
                "reading-comprehension",
                "tool-calling"
            ],
            "task_ids": ["tool-calling", "reading-comprehension"],
        },
    )

    # Test and add the card to the catalog
    test_card(card, strict=False)
    add_to_catalog(card, "cards.bfcl.simple_v3", overwrite=True)
