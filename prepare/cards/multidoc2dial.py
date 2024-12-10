from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    ExecuteExpression,
    ListFieldValues,
    Rename,
    Set,
)
from unitxt.settings_utils import get_settings
from unitxt.test_utils.card import test_card

settings = get_settings()

with settings.context(allow_unverified_code=True):
    card_abstractive = TaskCard(
        loader=LoadHF(path="multidoc2dial"),
        preprocess_steps=[
            Rename(
                field_to_field={"answers/text/0": "relevant_context"},
            ),
            ListFieldValues(fields=["utterance"], to_field="answers"),
            ExecuteExpression(
                expression="question.split('[SEP]')[0]", to_field="question"
            ),
            Set({"context_type": "document"}),
        ],
        task="tasks.qa.with_context.abstractive",
        templates="templates.qa.with_context.all",
        __tags__={
            "annotations_creators": "crowdsourced",
            "arxiv": "2109.12595",
            "language": "en",
            "language_creators": ["crowdsourced", "expert-generated"],
            "license": "apache-2.0",
            "multilinguality": "monolingual",
            "region": "us",
            "size_categories": ["10K<n<100K", "1K<n<10K", "n<1K"],
            "source_datasets": "extended|doc2dial",
            "task_categories": "question-answering",
            "task_ids": "open-domain-qa",
        },
        __description__=(
            "MultiDoc2Dial is a new task and dataset on modeling goal-oriented dialogues grounded in multiple documents. Most previous works treat document-grounded dialogue modeling as a machine reading comprehension task based on a single given document or passage. We aim to address more realistic scenarios where a goal-oriented information-seeking conversation involves multiple topics, and hence is grounded on different documents… See the full description on the dataset page: https://huggingface.co/datasets/multidoc2dial"
        ),
    )

    card_extractive = TaskCard(
        loader=LoadHF(path="multidoc2dial"),
        preprocess_steps=[
            Rename(
                field_to_field={"answers/text/0": "relevant_context"},
            ),
            ListFieldValues(fields=["relevant_context"], to_field="answers"),
            ExecuteExpression(
                expression="question.split('[SEP]')[0]", to_field="question"
            ),
            Set({"context_type": "document"}),
        ],
        task="tasks.qa.extractive",
        templates="templates.qa.with_context.all",
        __tags__={
            "annotations_creators": "crowdsourced",
            "arxiv": "2109.12595",
            "language": "en",
            "language_creators": ["crowdsourced", "expert-generated"],
            "license": "apache-2.0",
            "multilinguality": "monolingual",
            "region": "us",
            "size_categories": ["10K<n<100K", "1K<n<10K", "n<1K"],
            "source_datasets": "extended|doc2dial",
            "task_categories": "question-answering",
            "task_ids": "open-domain-qa",
        },
        __description__=(
            "MultiDoc2Dial is a new task and dataset on modeling goal-oriented dialogues grounded in multiple documents. Most previous works treat document-grounded dialogue modeling as a machine reading comprehension task based on a single given document or passage. We aim to address more realistic scenarios where a goal-oriented information-seeking conversation involves multiple topics, and hence is grounded on different documents… See the full description on the dataset page: https://huggingface.co/datasets/multidoc2dial"
        ),
    )

    for name, card in zip(
        ["abstractive", "extractive"], [card_abstractive, card_extractive]
    ):
        test_card(card)
        add_to_catalog(card, f"cards.multidoc2dial.{name}", overwrite=True)
