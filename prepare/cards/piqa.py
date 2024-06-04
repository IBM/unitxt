from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import ListFieldValues, RenameFields
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="piqa"),
    preprocess_steps=[
        ListFieldValues(fields=["sol1", "sol2"], to_field="choices"),
        RenameFields(
            field_to_field={"goal": "question", "label": "answer"},
        ),
    ],
    task="tasks.qa.multiple_choice.open",
    templates="templates.qa.multiple_choice.open.all",
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": ["1911.11641", "1907.10641", "1904.09728", "1808.05326"],
        "flags": ["croissant"],
        "language": "en",
        "language_creators": ["crowdsourced", "found"],
        "license": "unknown",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "question-answering",
        "task_ids": "multiple-choice-qa",
    },
    __description__=(
        "To apply eyeshadow without a brush, should I use a cotton swab or a toothpick?\n"
        "Questions requiring this kind of physical commonsense pose a challenge to state-of-the-art\n"
        "natural language understanding systems. The PIQA dataset introduces the task of physical commonsense reasoning\n"
        "and a corresponding benchmark dataset Physical Interaction: Question Answering or PIQA.\n"
        "Physical commonsense knowledge is a major challenge on the road to true AI-completeness,\n"
        "including robots that interact with the world and understand natural language.\n"
        "PIQA focuses on everyday situations with a preference for atypical solutions.\n"
        "The dataset is inspired by instructables.com, which provides users with instructions on how to build, craft,\n"
        "bake, or manipulate objects using everyday materials.\n"
        "The underlying task is formualted as multiple choice question answering:\n"
        "given a question `q` and two possible solutions `s1`, `s2`, a model or\n"
        "a human must choose the most appropriate solution, of which exactly one is correct.\n"
        "The dataset is further cleaned of basic artifacts using the AFLite algorithm which is an improvement of\n"
        "adversarial filtering. The dataset contains 16,000 examples for training, 2,000 for development and 3,000 for testing."
    ),
)
test_card(card, strict=False)
add_to_catalog(card, "cards.piqa", overwrite=True)
