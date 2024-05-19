from unitxt.blocks import LoadHF
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    AddFields,
    ListFieldValues,
    MapInstanceValues,
    RenameFields,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="super_glue", name="copa"),
    preprocess_steps=[
        "splitters.small_no_test",
        ListFieldValues(fields=["choice1", "choice2"], to_field="choices"),
        RenameFields(field_to_field={"premise": "context", "label": "answer"}),
        MapInstanceValues(
            mappers={
                "question": {  # https://people.ict.usc.edu/~gordon/copa.html
                    "cause": "What was the cause of this?",
                    "effect": "What happened as a result?",
                }
            }
        ),
        AddFields({"context_type": "sentence"}),
    ],
    task="tasks.qa.multiple_choice.with_context",
    templates="templates.qa.multiple_choice.with_context.all",
    __description__=(
        "SuperGLUE (https://super.gluebenchmark.com/) is a new benchmark styled after"
        "GLUE with a new set of more difficult language understanding tasks, improved"
        "resources, and a new public leaderboard."
        "The Choice Of Plausible Alternatives (COPA, Roemmele et al., 2011) dataset is a causal"
        "reasoning task in which a system is given a premise sentence and two possible alternatives. The"
        "system must choose the alternative which has the more plausible causal relationship with the premise."
        "The method used for the construction of the alternatives ensures that the task requires causal reasoning"
        "to solve. Examples either deal with alternative possible causes or alternative possible effects of the"
        "premise sentence, accompanied by a simple question disambiguating between the two instance"
        "types for the model. All examples are handcrafted and focus on topics from online blogs and a"
        "photography-related encyclopedia. Following the recommendation of the authors, we evaluate using"
        "accuracy."
    ),
    __tags__={
        "NLU": True,
        "annotations_creators": "expert-generated",
        "arxiv": "1905.00537",
        "croissant": True,
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
        "natural language understanding": True,
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "extended|other",
        "superglue": True,
        "task_categories": [
            "text-classification",
            "token-classification",
            "question-answering",
        ],
        "task_ids": [
            "natural-language-inference",
            "word-sense-disambiguation",
            "coreference-resolution",
            "extractive-qa",
        ],
    },
)

test_card(card, strict=False)
add_to_catalog(card, "cards.copa", overwrite=True)
