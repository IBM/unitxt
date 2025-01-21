import unitxt
from unitxt.blocks import LoadHF
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    ListFieldValues,
    MapInstanceValues,
    Rename,
    Set, Unique,
)
from unitxt.test_utils.card import test_card

with unitxt.settings.context(allow_unverified_code=True):
    card = TaskCard(
        loader=LoadHF(
            path="allenai/social_i_qa", data_classification_policy=["public"]
        ),
        preprocess_steps=[
            "splitters.small_no_test",
            Unique(['context', 'question', 'answerA', 'answerB', 'answerC']),
            ListFieldValues(
                fields=["answerA", "answerB", "answerC"], to_field="choices"
            ),
            MapInstanceValues(
                mappers={
                    "label": {
                        "1": 0,
                        "2": 1,
                        "3": 2,
                    }
                }
            ),
            Rename(field="label", to_field="answer"),
            Set({"context_type": "situation"}),
        ],
        task="tasks.qa.multiple_choice.with_context",
        templates="templates.qa.multiple_choice.with_context.all",
        __description__=(
            """Social IQa: Social Interaction QA, a question-answering benchmark for testing social commonsense intelligence. Contrary to many prior benchmarks that focus on physical or taxonomic knowledge, Social IQa focuses on reasoning about people's actions and their social implications. For example, given an action like "Jesse saw a concert" and a question like "Why did Jesse do this?", humans can easily infer that Jesse wanted "to see their favorite performer" or "to enjoy the music", and not "to see what's happening inside" or "to see if it works". The actions in Social IQa span a wide variety of social situations, and answer candidates contain both human-curated answers and adversarially-filtered machine-generated candidates. Social IQa contains over 37,000 QA pairs for evaluating models' abilities to reason about the social implications of everyday events and situations."""
        ),
        __tags__={
            "annotations_creators": "expert-generated",
            "arxiv": "1904.09728",
            "flags": ["NLU", "natural language understanding"],
            "language": "en",
            "language_creators": "other",
            "license": "other",
            "multilinguality": "monolingual",
            "region": "us",
            "size_categories": "10K<n<100K",
            "source_datasets": "extended|other",
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
    add_to_catalog(card, "cards.social_iqa", overwrite=True)
