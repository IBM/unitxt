from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.image_operators import ToImage
from unitxt.operators import Cast, Rename
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="lmms-lab/ai2d"),
    preprocess_steps=[
        ToImage(field="image", to_field="context"),
        Rename(field="options", to_field="choices"),
        Set(fields={"context_type": "image"}),
        Cast(field="answer", to="int"),
    ],
    task="tasks.qa.multiple_choice.with_context[metrics=[metrics.exact_match_mm]]",
    templates="templates.qa.multiple_choice.with_context.no_intro.all",
    __tags__={},
    __description__=(
        "AI2 Diagrams (AI2D) is a dataset of over 5000 grade school science diagrams with over 150000 rich annotations, their ground truth syntactic parses, and more than 15000 corresponding multiple choice questions."
    ),
)

test_card(card)
add_to_catalog(card, "cards.ai2d", overwrite=True)
