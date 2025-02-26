from typing import Any, Dict, Optional

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard
from unitxt.collections_operators import Wrap
from unitxt.loaders import LoadHF
from unitxt.operators import Copy, InstanceOperator, Set
from unitxt.splitters import SplitRandomMix
from unitxt.string_operators import Join
from unitxt.templates import InputOutputTemplate


# TODO - To be removed.
class Breakpoint(InstanceOperator):
    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        return instance
# Benchmark
card = TaskCard(
    loader=LoadHF(
        path="hotpotqa/hotpot_qa",
        name="distractor",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        SplitRandomMix(
            {
                "test": "train[30%]",
                "train": "train[70%]",
            }),
        Copy(
            field_to_field={
                "question": "question",
                "id": "question_id",
                "level": "metadata_field/level"
            },
        ),
        Copy(
            field="context/title",
            to_field="reference_context_ids",
        ),
        Join(
            field="context/sentences",
            by=" ",
            to_field="reference_contexts",
            process_every_value=True,
        ),
        Set(
            fields={
                "is_answerable_label": True,
            }
        ),
        Wrap(
            field="answer",
            inside="list",
            to_field="reference_answers",
        ),
    ],
    task="tasks.rag.end_to_end",
    templates={"default": "templates.rag.end_to_end.json_predictions"},
    __tags__={"license": "CC BY-SA 4.0"},
    __description__="""TODO
""",
)
wrong_answer = {
    "contexts": ["hi"],
    "is_answerable": True,
    "answer": "Don't know",
    "context_ids": ["id0"],
}

# test_card(
#     card,
#     strict=True,
#     full_mismatch_prediction_values=[json.dumps(wrong_answer)],
#     debug=False,
#     demos_taken_from="test",
#     demos_pool_size=5,
# )

add_to_catalog(card, "cards.rag.benchmark.hotpotqa.en", overwrite=True)


# Documents
card = TaskCard(
    loader=LoadHF(
        path="hotpotqa/hotpot_qa",
        name="distractor",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        Copy(field="id", to_field="document_id"),
        # TODO This line is wrong
        Wrap(field="passage", inside="list", to_field="passages"),
        Set(
            fields={
                "metadata_field": "",
                "title": "", # TODO
            }
        ),
        # TODO SET TITLE TO BE DOCUMENT ID -
    ],
    task="tasks.rag.corpora",
    templates={
        "empty": InputOutputTemplate(
            input_format="",
            output_format="",
        ),
    },
    __tags__={"license": "CC BY-SA 4.0"},
    __description__="""TODO
""",
)

# Not testing card, because documents are not evaluated.
add_to_catalog(card, "cards.rag.documents.hotpotqa.en", overwrite=True)
