from unitxt.blocks import LoadHF, RenameFields, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import ListFieldValues
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="Muennighoff/babi"),
    preprocess_steps=[
        RenameFields(field_to_field={"passage": "context"}),
        Set({"context_type": "description"}),
        ListFieldValues(fields=["answer"], to_field="answers"),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
    __tags__={"region": "us"},
    __description__=(
        "Creation (Copied & adapted from https://github.com/stanford-crfm/helm/blob/0eaaa62a2263ddb94e9850ee629423b010f57e4a/src/helm/benchmark/scenarios/babi_qa_scenario.py):\n"
        "!wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n"
        "!tar -xf tasks_1-20_v1-2.tar.gz\n"
        "import json\n"
        "from typing import List\n"
        "tasks = list(range(1, 20))\n"
        'splits = ["train", "valid", "test"]\n'
        "def process_path(path: str) -> str:\n"
        '"""Turn a path string (task 19) from the original format \'s,w\' to a verbal… See the full description on the dataset page: https://huggingface.co/datasets/Muennighoff/babi.'
    ),
)

test_card(card)
add_to_catalog(card, "cards.babi.qa", overwrite=True)
