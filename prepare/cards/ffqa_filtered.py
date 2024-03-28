from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import (
    AddFields,
    CopyFields,
    ExecuteExpression,
    FilterByCondition,
    ListFieldValues,
)
from src.unitxt.test_utils.card import test_card

"""Filtered version of the WikiQA-Free_Form_QA dataset.
If you would like to use the full dataset, please copy and modify this card as ffqa.py.
"""

# Dataset structure:
# DatasetDict({
#     2k: Dataset({
#         features: ['conversations'],
#         num_rows: 600
#     })
#     4k: Dataset({
#         features: ['conversations'],
#         num_rows: 600
#     })
#     8k: Dataset({
#         features: ['conversations'],
#         num_rows: 600
#     })
#     16k: Dataset({
#         features: ['conversations'],
#         num_rows: 600
#     })
# })
#
# conversations: [
#     {
#         "from": "human",
#         "tok_len": int,
#         "value": str,
#     },
#     {
#         "from": "agent",
#         "tok_len": int,
#         "value": str,
#     },
# ]
#
# value (There is also a pattern Document and Question are reversed):
#   Answer the question based on the information provided in the document given below. The answer should be a single word or a number or a short phrase of few words
#
#   Document:
#   <document>
#
#   Question:
#   <question>
#


# TODO: Remove duplicate data


# Some of the data is longer than the name of the split.
# For example, it may be 9000 tokens even if it is included in an 8k split.
# Set an upper limit to exclude sentences that are too long.
# NOTE: Only 8k split has been adjusted for now.
token_limit_map = {
    "2k": 2048,
    "4k": 4096,
    "8k": 8800,
    "16k": 16384,
}


def add_card(split: str):
    card = TaskCard(
        loader=LoadHF(path="abacusai/WikiQA-Free_Form_QA"),
        preprocess_steps=[
            CopyFields(
                field_to_field={
                    "conversations/0/value": "inputs",
                    "conversations/0/tok_len": "inputs_len",
                    "conversations/1/value": "answer",
                },
            ),
            ListFieldValues(fields=["answer"], to_field="answers"),
            FilterByCondition(
                values={"inputs_len": token_limit_map[split]},
                condition="lt",
            ),
            ExecuteExpression(
                expression='re.search(r"Document:\\s(.*)(\\n\\n|$)", inputs).group(1)',
                imports_list=["re"],
                to_field="context",
            ),
            ExecuteExpression(
                expression='re.search(r"Question:\\s(.*)(\\n\\n|$)", inputs).group(1)',
                imports_list=["re"],
                to_field="question",
            ),
            AddFields({"context_type": "document"}),
            SplitRandomMix(
                {
                    "train": f"{split}[80%]",
                    "validation": f"{split}[10%]",
                    "test": f"{split}[10%]",
                }
            ),
        ],
        task="tasks.qa.with_context.extractive",
        templates="templates.qa.with_context.all",
    )

    test_card(card)
    add_to_catalog(card, f"cards.ffqa_filtered.{split}", overwrite=True)


for split in ["2k", "4k", "8k", "16k"]:
    add_card(split)
