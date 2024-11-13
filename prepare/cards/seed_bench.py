from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.image_operators import ToImage, ToRGB
from unitxt.operators import ListFieldValues, MapValues
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="lmms-lab/SEED-Bench"),
    preprocess_steps=[
        ToImage(field="image", to_field="context", process_every_value=True),
        ToRGB(field="context", process_every_value=True),
        ListFieldValues(
            fields=["choice_a", "choice_b", "choice_c", "choice_d"], to_field="choices"
        ),
        Set(fields={"context_type": "video"}),
        MapValues(mapping={"A": 0, "B": 1, "C": 2, "D": 3}, field="answer"),
    ],
    task="tasks.qa.multiple_choice.with_context",
    templates="templates.qa.multiple_choice.with_context.no_intro.all",
    __tags__={},
    __description__=(
        "SEED-Bench-1 consists of 19K multiple-choice questions with accurate human annotations, covering 12 evaluation dimensions including both the spatial and temporal understanding."
    ),
)

test_card(card)
add_to_catalog(card, "cards.seed_bench", overwrite=True)
