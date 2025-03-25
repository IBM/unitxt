from unitxt import get_from_catalog
from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.image_operators import ToImage
from unitxt.operators import Cast, Rename, Shuffle
from unitxt.test_utils.card import test_card


templates = get_from_catalog("templates.qa.multiple_choice.with_context.no_intro.all")
default_template = get_from_catalog("templates.qa.multiple_choice.with_context.ai2d")

card = TaskCard(
    loader=LoadHF(path="lmms-lab/ai2d"),
    preprocess_steps=[
        Shuffle(),
        ToImage(field="image", to_field="context"),
        Rename(field="options", to_field="choices"),
        Set(fields={"context_type": "image"}),
        Cast(field="answer", to="int"),
    ],
    task="tasks.qa.multiple_choice.with_context[metrics=[metrics.exact_match_mm]]",
    templates=[default_template, *templates.items],
    __tags__={},
    __description__=(
        "AI2 Diagrams (AI2D) is a dataset of over 5000 grade school science diagrams with over 150000 rich annotations, their ground truth syntactic parses, and more than 15000 corresponding multiple choice questions."
    ),
)

test_card(card, strict=False)
add_to_catalog(card, "cards.ai2d", overwrite=True)
