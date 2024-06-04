import os

from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    Task,
    TaskCard,
    TemplatesList,
)
from unitxt.operators import JoinStr
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="mbpp", name="full", split="test"),
    preprocess_steps=[
        JoinStr(field_to_field={"test_list": "test_list_str"}, separator=os.linesep),
    ],
    task=Task(
        inputs=["text", "test_list_str"],
        outputs=["test_list", "code"],
        metrics=["metrics.bleu"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format='"""{text}\n\n{test_list_str}"""',
                output_format="{code}",
            ),
        ]
    ),
    __tags__={
        "annotations_creators": ["crowdsourced", "expert-generated"],
        "arxiv": "2108.07732",
        "flags": ["code-generation", "croissant"],
        "language": "en",
        "language_creators": ["crowdsourced", "expert-generated"],
        "license": "cc-by-4.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "n<1K",
        "source_datasets": "original",
        "task_categories": "text2text-generation",
    },
    __description__=(
        "Dataset Card for Mostly Basic Python Problems (mbpp) Dataset Summary The benchmark consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases. As described in the paper, a subset of the data has been hand-verified by us. Released here as part… See the full description on the dataset page: https://huggingface.co/datasets/mbpp."
    ),
)

test_card(
    card,
    demos_taken_from="test",
    demos_pool_size=1,
    num_demos=0,
    strict=False,
    loader_limit=500,
    debug=False,
)
add_to_catalog(card, "cards.mbpp", overwrite=True)
